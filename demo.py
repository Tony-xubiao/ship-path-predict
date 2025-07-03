import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Nadam
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False

# 这个demo里，使用过去的'longitude', 'latitude', 'speed', 'course' 数据，预测'longitude', 'latitude'两个参数，“过去”的长度取决于参数n_steps，只能预测下一秒数据


# 1. 数据预处理
def preprocess_data(data_path):
    # 读取已处理好的等间隔数据
    data = pd.read_csv(data_path, parse_dates=['timestamp'])
    data = data.sort_values('timestamp')  # 确保时间顺序正确

    # 直接使用原始数据（假设已经是1秒间隔）
    features = data[['longitude', 'latitude', 'speed', 'course']]

    # Min-Max归一化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)

    return scaler, scaled_data


# 2. 构建监督学习序列（保持不变）
def create_sequences(data, n_steps=5):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps, :2])  # 预测下一时刻的经度纬度
    return np.array(X), np.array(y)


# 3. LSTM模型构建（保持不变）
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(2))  # 输出经度和纬度
    optimizer = Nadam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model


# 4. 主程序（修改数据预处理调用）
if __name__ == "__main__":
    # 参数设置
    n_steps = 5  # 基于过去5秒预测下一秒
    batch_size = 100
    epochs = 20

    # 数据预处理（移除时间间隔参数）
    scaler, processed_data = preprocess_data('./data/wave_ship_data.csv')

    # 创建序列数据集
    X, y = create_sequences(processed_data, n_steps)

    # 划分训练集和测试集（4:1）
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 构建模型
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    # 训练模型
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        verbose=1)

    # 预测与评估（保持不变）
    y_pred = model.predict(X_test)


    # 构造完整的归一化数据矩阵（包含所有特征）
    dummy_test = np.zeros((len(y_test), processed_data.shape[1]))
    dummy_test[:, :2] = y_test  # 填充真实的归一化经纬度标签
    dummy_test[:, 2:] = X_test[:, -1, 2:]  # 使用输入序列最后一刻的归一化速度和航向

    # 反归一化
    y_test_actual = scaler.inverse_transform(dummy_test)[:, :2]

    # 同理处理预测值
    dummy_pred = np.zeros((len(y_pred), processed_data.shape[1]))
    dummy_pred[:, :2] = y_pred
    dummy_pred[:, 2:] = X_test[:, -1, 2:]  # 与预测对应的输入序列末尾特征
    y_pred_actual = scaler.inverse_transform(dummy_pred)[:, :2]

    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    print(f'Test RMSE: {rmse:.6f} degrees')

    # 可视化轨迹对比
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual[:, 0], y_test_actual[:, 1], 'b-', label='Actual Track')
    plt.plot(y_pred_actual[:, 0], y_pred_actual[:, 1], 'r--', label='Predicted Track')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Ship Trajectory Prediction (1s interval data)')
    plt.legend()
    plt.show()
