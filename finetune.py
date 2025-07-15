# 加载现有的已经训练好的模型继续训练（finetune）

import os
import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import mean_squared_error
from keras.optimizers import Nadam

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False


def load_latest_model_and_scalers(checkpoint_dir='checkpoint', time_str=""):
    """加载最新的模型和对应的scalers"""
    # 查找最新的模型文件
    model_files = glob.glob(os.path.join(checkpoint_dir, 'model_*.h5'))
    if not model_files:
        raise FileNotFoundError("未找到该model文件，请check文件名")

    # 获取模型文件
    model = load_model(os.path.join(checkpoint_dir, f'model_{time_str}.h5'),
                       compile=False)
    model.compile(optimizer=Nadam(), loss='mse')

    # 加载对应的scalers
    scaler_files = glob.glob(os.path.join(checkpoint_dir, f'scaler_{time_str}_*.pkl'))
    if not scaler_files:
        raise FileNotFoundError(f"找不到与模型model_{time_str}.h5对应的scaler文件")

    scalers = {}
    for scaler_file in scaler_files:
        col = scaler_file.split('_')[-1].split('.')[0]
        scalers[col] = joblib.load(scaler_file)

    return model, scalers, time_str


def preprocess_new_data(scalers, data_path):
    """使用已有的scalers预处理新数据"""
    data = pd.read_csv(data_path, parse_dates=['timestamp'])
    data = data.sort_values('timestamp')
    features = data[['longitude', 'latitude', 'speed', 'course']]

    # 使用已有的scalers进行变换
    scaled_features = []
    for col in features.columns:
        if col in scalers:
            scaled_col = scalers[col].transform(features[[col]])
            scaled_features.append(scaled_col.ravel())
        else:
            raise ValueError(f"找不到特征 {col} 的scaler")

    scaled_data = np.column_stack(scaled_features)
    return scaled_data


def create_sequences(data, n_steps=60, forecast_steps=10):
    X, y = [], []
    for i in range(len(data) - n_steps - forecast_steps + 1):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps: i + n_steps + forecast_steps, :2])  # 仅取经纬度
    return np.array(X), np.array(y)


def inverse_scale_coordinates(scalers, data):
    """使用独立的scaler进行逆变换"""
    lon_scaler = scalers['longitude']
    lat_scaler = scalers['latitude']

    lon = data[..., 0].reshape(-1, 1)
    inverted_lon = lon_scaler.inverse_transform(lon)

    lat = data[..., 1].reshape(-1, 1)
    inverted_lat = lat_scaler.inverse_transform(lat)

    inverted = np.empty_like(data)
    inverted[..., 0] = inverted_lon.reshape(data.shape[:-1])
    inverted[..., 1] = inverted_lat.reshape(data.shape[:-1])
    return inverted


def plot_comparison(idx, y_test_real, y_pred_real):
    """绘制单个样本的预测对比"""
    actual = y_test_real[idx]
    predicted = y_pred_real[idx]

    plt.figure(figsize=(16, 6))
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


if __name__ == "__main__":
    n_steps = 60
    forecast_steps = 10
    batch_size = 64
    epochs = 5  # 微调时的epoch数

    # 1. 加载指定的模型和对应的scalers
    print("加载指定模型和scalers...")
    time_str = "2025-07-15-10-29-55"
    checkpoint_dir = "checkpoint"
    # 期望的model_name是"model_{time_str}.h5"，scalers是"scaler_{time_str}_{col}.pkl"

    model, scalers, original_time_str = load_latest_model_and_scalers(checkpoint_dir=checkpoint_dir, time_str=time_str)

    # 2. 归一化预处理新数据（使用指定的scalers）
    print("预处理新数据...")
    new_data_path = './data/wave_ship_data_copy.csv'
    processed_new_data = preprocess_new_data(scalers, new_data_path)

    # 3. 创建新数据集
    print("创建序列数据...")
    X_new, y_new = create_sequences(processed_new_data, n_steps, forecast_steps)

    # 4. 划分数据集
    split = int(0.8 * len(X_new))
    X_train_new, X_test_new = X_new[:split], X_new[split:]
    y_train_new, y_test_new = y_new[:split], y_new[split:]

    # 5. 继续训练模型
    print("开始微调（继续训练）模型...")
    history = model.fit(X_train_new, y_train_new,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test_new, y_test_new),
                        verbose=1)

    # 6. 保存微调后的模型和scalers
    now = datetime.now()
    time_str = f"{now.year}-{now.month:02d}-{now.day:02d}-{now.hour:02d}-{now.minute:02d}-{now.second:02d}"
    save_dir = os.path.join("checkpoint", time_str)

    # 保存模型时注明是微调版本
    model_name = f"{save_dir}/model_finetuned_{original_time_str}_to_{time_str}.h5"
    model.save(model_name)

    # 保存scalers（使用原始时间戳，因为scalers没有变化）
    os.makedirs(save_dir, exist_ok=True)
    for col, scaler in scalers.items():
        joblib.dump(scaler, f"{save_dir}/scaler_finetuned_{original_time_str}_{col}.pkl")

    print(f"微调完成，模型和scalers已保存至{save_dir}目录\n原始模型时间: {original_time_str}\n微调时间: {time_str}")

    # 7. 评估微调后的模型
    print("\n评估微调后的模型...")
    y_pred_new = model.predict(X_test_new)

    # 逆变换得到真实坐标
    y_test_real = inverse_scale_coordinates(scalers, y_test_new)
    y_pred_real = inverse_scale_coordinates(scalers, y_pred_new)

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
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test_new), 3, replace=False)
    print("\n生成可视化对比...")
    for idx in sample_indices:
        plot_comparison(idx, y_test_real, y_pred_real)
