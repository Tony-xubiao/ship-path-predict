import os
import sys
import time
from io import StringIO

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Model
from keras.optimizers import Nadam
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from shipai.model.model_load import get_path, train_data_path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False

# 新增：数据平滑处理函数
def smooth_data(data, window_size=5, method='moving_average'):
    """
    数据平滑处理
    参数:
        data: 输入数据 (numpy数组)
        window_size: 平滑窗口大小
        method: 平滑方法 ('moving_average', 'exponential', 'savitzky_golay')
    返回:
        平滑后的数据
    """
    smoothed_data = np.zeros_like(data)
    
    if method == 'moving_average':
        # 简单移动平均
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            smoothed_data[i] = np.mean(data[start:end], axis=0)
    
    elif method == 'exponential':
        # 指数平滑
        alpha = 2 / (window_size + 1)
        smoothed_data[0] = data[0]
        for i in range(1, len(data)):
            smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i-1]
    
    elif method == 'savitzky_golay':
        # Savitzky-Golay 滤波器
        for col in range(data.shape[1]):
            smoothed_data[:, col] = savgol_filter(data[:, col], window_size, 2)
    
    return smoothed_data

# 1. 数据预处理（增加平滑处理）
def preprocess_data(sftp, data_path, smooth_window=5, smooth_method='savitzky_golay'):
    try:
        with sftp.open(data_path, 'r') as f:
            content = f.read().decode('utf-8')
            data = pd.read_csv(StringIO(content), parse_dates=['timestamp'])
    except Exception as e:
        print(f"读取 CSV 数据出错，请检查数据，错误原因: {e}")
        sys.exit(1)

    data = data.sort_values('timestamp')
    features = data[['longitude', 'latitude', 'speed', 'course']].values
    
    # 对原始数据进行平滑处理
    smoothed_features = smooth_data(features, window_size=smooth_window, method=smooth_method)
    
    # 初始化不同特征的Scaler
    scaler_lon_lat = MinMaxScaler()  # 经纬度共用
    scaler_speed = MinMaxScaler()    # 速度单独
    scaler_course = MinMaxScaler()   # 航向单独
    
    # 分别进行归一化
    lon_lat_scaled = scaler_lon_lat.fit_transform(smoothed_features[:, :2])
    speed_scaled = scaler_speed.fit_transform(smoothed_features[:, 2].reshape(-1, 1))
    course_scaled = scaler_course.fit_transform(smoothed_features[:, 3].reshape(-1, 1))
    
    # 合并归一化后的数据
    scaled_data = np.hstack((lon_lat_scaled, speed_scaled, course_scaled))
    
    return scaler_lon_lat, scaler_speed, scaler_course, scaled_data, features, smoothed_features, data

# 2. 构建监督学习序列
def create_sequences(data, n_steps=5):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps, :4])  # 预测四个特征
    return np.array(X), np.array(y)

# 自适应学习率回调
def adaptive_lr():
    return ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-6,
        verbose=1
    )

def lr_schedule(epoch):
    if epoch < 1000:
        return 0.0001
    else:
        return 0.00005

# 3. LSTM模型构建
def build_simple_model(input_shape, output_steps=4, units=128):
    inputs = Input(shape=input_shape)
    x = LSTM(units, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(units)(x)
    x = Dropout(0.3)(x)
    outputs = Dense(output_steps)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Nadam(0.0001), loss='mse')
    return model

def save_model_and_scalers(sftp, mmsi, model, scalers, model_code):
    remote_path = get_path(mmsi, model_code)
    timestamp = int(time.time())
    """保存模型和scaler到指定目录"""
    model_local = os.path.join('checkpoint', f'model_{timestamp}.h5')
    model.save(model_local)
    sftp.put(model_local, f'{remote_path}/model.h5')
    os.remove(model_local)
    print(f"Model saved to {remote_path}")

    # 保存scaler
    scaler_names = ['scaler_lon_lat', 'scaler_speed', 'scaler_course']
    for name, scaler in zip(scaler_names, scalers):
        scaler_local = os.path.join('checkpoint', f'scaler_{name}_{timestamp}.pkl')
        joblib.dump(scaler, scaler_local)
        sftp.put(scaler_local, f'{remote_path}/{name}.pkl')
        os.remove(scaler_local)
    print(f"scalers saved to {remote_path}")


def train(sftp, mmsi, model_code):
    print(f"进入到模型训练方法...mmsi:{mmsi}___modelCode:{model_code}")
    # 参数设置
    n_steps = 20
    batch_size = 32
    epochs = 500
    smooth_window = 7  # 平滑窗口大小
    smooth_method = 'savitzky_golay'  # 平滑方法: 'moving_average', 'exponential', 'savitzky_golay'
    random_state = 42  # 随机种子确保可重复性

    data_file_path = train_data_path(mmsi, model_code)
    try:
        sftp.stat(data_file_path)
    except FileNotFoundError:
        raise Exception("请提供有效的数据文件路径")

    print(f'读取到数据文件{data_file_path}，开始训练模型...')

    # 数据预处理（获取多个Scaler）
    scaler_lon_lat, scaler_speed, scaler_course, processed_data, raw_features, smoothed_features, data_df = preprocess_data(
        sftp,
        data_file_path,
        smooth_window=smooth_window,
        smooth_method=smooth_method
    )

    # 创建序列数据集
    X, y = create_sequences(processed_data, n_steps)

    # 划分训练集和测试集 - 使用随机划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=random_state
    )

    # 构建并训练模型
    model = build_simple_model((X_train.shape[1], X_train.shape[2]))
    callbacks = [
        LearningRateScheduler(lr_schedule),
        EarlyStopping(monitor='val_loss', patience=50, min_delta=0.0001, restore_best_weights=True)
    ]

    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.2,
              verbose=1,
              callbacks=callbacks)

    print(f'模型训练结束，开始保存模型和scaler...')

    # 保存模型和scaler
    save_model_and_scalers(sftp, mmsi, model, [scaler_lon_lat, scaler_speed, scaler_course], model_code)

    # 常规预测与评估
    y_pred = model.predict(X_test).squeeze()

    # 反归一化处理（仅经纬度）
    y_pred_lonlat = scaler_lon_lat.inverse_transform(y_pred[:, :2])
    y_test_lonlat = scaler_lon_lat.inverse_transform(y_test[:, :2])

    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_test_lonlat, y_pred_lonlat))
    print(f'Test RMSE: {rmse:.6f} degrees')
    return X.shape[0]