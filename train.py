# 用过去n_steps个时间步的'longitude', 'latitude', 'speed', 'course'数据，预测将来forecast_steps个时间步的'longitude', 'latitude'数据
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape
from keras.optimizers import Nadam
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False

def preprocess_data(data_path):
    data = pd.read_csv(data_path, parse_dates=['timestamp'])
    data = data.sort_values('timestamp')
    features = data[['longitude', 'latitude', 'speed', 'course']]
    # 为每个特征创建独立的scaler
    scalers = {}
    scaled_features = []
    for col in features.columns:
        scaler = MinMaxScaler()
        scaled_col = scaler.fit_transform(features[[col]])
        scalers[col] = scaler
        scaled_features.append(scaled_col.ravel())  # 转换为一维数组

    # 合并所有特征的缩放结果
    scaled_data = np.column_stack(scaled_features)
    return scalers, scaled_data


def create_sequences(data, n_steps=60, forecast_steps=10):
    X, y = [], []
    for i in range(len(data) - n_steps - forecast_steps + 1):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps: i + n_steps + forecast_steps, :2])  # 仅取经纬度
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(10 * 2))
    model.add(Reshape((10, 2)))
    optimizer = Nadam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model


def save_scalers(scalers,save_dir, time_str):
    """单独保存每个特征的scaler"""
    os.makedirs(save_dir, exist_ok=True)
    for col, scaler in scalers.items():
        joblib.dump(scaler, f"{save_dir}/scaler_{time_str}_{col}.pkl")


def inverse_scale_coordinates(scalers, data):
    """使用独立的scaler进行逆变换"""
    # 获取对应的scaler
    lon_scaler = scalers['longitude']
    lat_scaler = scalers['latitude']

    # 处理经度维度
    lon = data[..., 0].reshape(-1, 1)
    inverted_lon = lon_scaler.inverse_transform(lon)

    # 处理纬度维度
    lat = data[..., 1].reshape(-1, 1)
    inverted_lat = lat_scaler.inverse_transform(lat)

    # 重组数据
    inverted = np.empty_like(data)
    inverted[..., 0] = inverted_lon.reshape(data.shape[:-1])
    inverted[..., 1] = inverted_lat.reshape(data.shape[:-1])
    return inverted


if __name__ == "__main__":
    n_steps = 60
    forecast_steps = 10
    batch_size = 64
    epochs = 5
    # 数据预处理（返回字典形式的scalers）
    scalers, processed_data = preprocess_data('./data/wave_ship_data.csv')

    # 创建数据集
    X, y = create_sequences(processed_data, n_steps, forecast_steps)

    # 划分数据集
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 构建并训练模型
    model = build_lstm_model((n_steps, X_train.shape[2]))
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_test, y_test),
              verbose=1)

    # 保存模型和scaler（修改保存方式）
    now = datetime.now()
    time_str = f"{now.year}-{now.month:02d}-{now.day:02d}-{now.hour:02d}-{now.minute:02d}-{now.second:02d}"
    print(time_str)

    model_name = "./checkpoint/model_" + time_str + ".h5"
    model.save(model_name)
    save_dir = "checkpoint"
    save_scalers(scalers, save_dir, time_str)  # 新增scaler保存方式
    print("训练完成，模型和scaler已保存至checkpoint目录")

    # ============ 新增部分：模型评估与可视化 ============
    # 预测测试集
    y_pred = model.predict(X_test)

    # 逆变换得到真实坐标
    y_test_real = inverse_scale_coordinates(scalers, y_test)
    y_pred_real = inverse_scale_coordinates(scalers, y_pred)

    # 计算MSE
    total_mse = mean_squared_error(
        y_test_real.reshape(-1, 2),
        y_pred_real.reshape(-1, 2)
    )
    print(f"\n整体测试集MSE: {total_mse:.6f}")

    # 分时间步计算MSE
    print("\n各时间步MSE：")
    for step in range(forecast_steps):
        step_mse = mean_squared_error(
            y_test_real[:, step, :],
            y_pred_real[:, step, :]
        )
        print(f"步骤 {step + 1}: {step_mse:.6f}")

    # 可视化对比
    def plot_comparison(idx):
        """绘制单个样本的预测对比"""
        actual = y_test_real[idx]
        predicted = y_pred_real[idx]

        plt.figure(figsize=(16, 6))

        # 子图1：轨迹对比
        # plt.subplot(1, 2, 1)
        plt.plot(actual[:, 0], actual[:, 1], 'b-o', label='真实轨迹', markersize=5)
        plt.plot(predicted[:, 0], predicted[:, 1], 'r--x', label='预测轨迹', markersize=5)
        plt.xlabel('经度')
        plt.ylabel('纬度')
        plt.title(
            f'样本 {idx} 轨迹对比\n(起点距离：{np.sqrt((actual[0, 0] - predicted[0, 0]) ** 2 + (actual[0, 1] - predicted[0, 1]) ** 2):.4f}°)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    # 随机选择3个样本可视化
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), 3, replace=False)
    print("\n生成可视化对比...")
    for idx in sample_indices:
        plot_comparison(idx)
