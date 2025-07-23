import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.optimizers import Nadam
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
import joblib
import os
from datetime import datetime
from scipy.signal import savgol_filter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False

# 数据平滑处理函数（与train.py一致）
def smooth_data(data, window_size=5, method='moving_average'):
    smoothed_data = np.zeros_like(data)
    
    if method == 'moving_average':
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            smoothed_data[i] = np.mean(data[start:end], axis=0)
    
    elif method == 'exponential':
        alpha = 2 / (window_size + 1)
        smoothed_data[0] = data[0]
        for i in range(1, len(data)):
            smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i-1]
    
    elif method == 'savitzky_golay':
        for col in range(data.shape[1]):
            smoothed_data[:, col] = savgol_filter(data[:, col], window_size, 2)
    
    return smoothed_data

# 创建序列数据（与train.py一致）
def create_sequences(data, n_steps=5):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps, :4])  # 预测四个特征
    return np.array(X), np.array(y)

# 迭代预测函数（与train.py一致）
def iterative_predict(model, initial_seq, scaler_lon_lat, predict_steps=30):
    current_seq = initial_seq.copy()
    predicted = []
    
    for _ in range(predict_steps):
        pred = model.predict(current_seq[np.newaxis, ...])  # 输出可能是 3D
        pred = pred.reshape(-1)  # 强制展平为 1D
        predicted.append(pred)
        current_seq = np.vstack([current_seq[1:], pred])
    
    predicted = np.array(predicted)
    predicted_lonlat = scaler_lon_lat.inverse_transform(predicted[:, :2])
    return predicted_lonlat

# 自适应学习率回调（与train.py一致）
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

# 保存模型和scalers（与train.py一致）
def save_model_and_scalers(model, scalers, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.save(os.path.join(save_dir, 'model.h5'))
    
    scaler_names = ['scaler_lon_lat', 'scaler_speed', 'scaler_course']
    for name, scaler in zip(scaler_names, scalers):
        joblib.dump(scaler, os.path.join(save_dir, f'{name}.pkl'))
    
    print(f"Model and scalers saved to {save_dir}")

def finetune(model_dir, new_data_path, validate_data_path='./data/validate_data.csv'):
    # 参数设置（与train.py保持一致）
    n_steps = 20
    batch_size = 32
    epochs = 50
    predict_steps = 5
    smooth_window = 7
    smooth_method = 'savitzky_golay'
    random_state = 42

    # 1. 加载已有模型和scalers
    print(f"Loading model and scalers from {model_dir}")
    model = load_model(os.path.join(model_dir, 'model.h5'),compile=False)
    model.compile(optimizer=Nadam(), loss='mse')
    scaler_lon_lat = joblib.load(os.path.join(model_dir, 'scaler_lon_lat.pkl'))
    scaler_speed = joblib.load(os.path.join(model_dir, 'scaler_speed.pkl'))
    scaler_course = joblib.load(os.path.join(model_dir, 'scaler_course.pkl'))

    # 2. 加载新数据并进行预处理
    print(f"Loading and preprocessing new data from {new_data_path}")
    new_data = pd.read_csv(new_data_path, parse_dates=['timestamp'])
    new_data = new_data.sort_values('timestamp')
    new_features = new_data[['longitude', 'latitude', 'speed', 'course']].values
    
    # 对新数据进行平滑处理
    smoothed_features = smooth_data(new_features, window_size=smooth_window, method=smooth_method)
    
    # 使用已有scaler进行归一化
    lon_lat_scaled = scaler_lon_lat.transform(smoothed_features[:, :2])
    speed_scaled = scaler_speed.transform(smoothed_features[:, 2].reshape(-1, 1))
    course_scaled = scaler_course.transform(smoothed_features[:, 3].reshape(-1, 1))
    scaled_data = np.hstack((lon_lat_scaled, speed_scaled, course_scaled))

    # 3. 创建序列数据集
    X_new, y_new = create_sequences(scaled_data, n_steps)

    # 4. 微调模型
    print("Starting fine-tuning...")
    callbacks = [
        LearningRateScheduler(lr_schedule),
        EarlyStopping(monitor='val_loss', patience=50, min_delta=0.0001, restore_best_weights=True)
    ]

    history = model.fit(
        X_new, y_new,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1,
        callbacks=callbacks
    )

    # 5. 保存微调后的模型和scalers
    now = datetime.now()
    timestamp = f"{now.year}-{now.month:02d}-{now.day:02d}-{now.hour:02d}-{now.minute:02d}-{now.second:02d}"
    save_dir = os.path.join('./checkpoint', f"finetuned_{timestamp}")
    save_model_and_scalers(model, [scaler_lon_lat, scaler_speed, scaler_course], save_dir)

    # 6. 评估微调后的模型
    y_pred = model.predict(X_new).squeeze()
    y_pred_lonlat = scaler_lon_lat.inverse_transform(y_pred[:, :2])
    y_new_lonlat = scaler_lon_lat.inverse_transform(y_new[:, :2])
    
    rmse = np.sqrt(mean_squared_error(y_new_lonlat, y_pred_lonlat))
    print(f'Fine-tuned model RMSE on new data: {rmse:.6f} degrees')

    # 7. 使用validate_data.csv进行验证和可视化（与train.py风格一致）
    print("Validating with validate_data.csv...")
    validate_data = pd.read_csv(validate_data_path, parse_dates=['timestamp'])
    validate_data = validate_data.sort_values('timestamp')
    validate_features = validate_data[['longitude', 'latitude', 'speed', 'course']].values
    
    # 对验证数据进行平滑处理
    validate_smoothed = smooth_data(validate_features, window_size=smooth_window, method=smooth_method)
    
    # 使用已有scaler进行归一化
    validate_lonlat = scaler_lon_lat.transform(validate_smoothed[:, :2])
    validate_speed = scaler_speed.transform(validate_smoothed[:, 2].reshape(-1, 1))
    validate_course = scaler_course.transform(validate_smoothed[:, 3].reshape(-1, 1))
    validate_scaled = np.hstack((validate_lonlat, validate_speed, validate_course))
    
    # 创建验证序列
    X_val, y_val = create_sequences(validate_scaled, n_steps)
    
    # 预测验证集
    y_val_pred = model.predict(X_val).squeeze()
    y_val_pred_lonlat = scaler_lon_lat.inverse_transform(y_val_pred[:, :2])
    y_val_lonlat = scaler_lon_lat.inverse_transform(y_val[:, :2])
    
    # 迭代预测 - 随机选择一段序列进行多步预测
    start_idx = np.random.randint(0, len(validate_scaled) - n_steps - predict_steps)
    initial_sequence = validate_scaled[start_idx:start_idx+n_steps]
    predicted_coords = iterative_predict(model, initial_sequence, scaler_lon_lat, predict_steps)
    
    # 获取对应的实际坐标
    actual_start_idx = start_idx + n_steps
    actual_end_idx = min(actual_start_idx + predict_steps, len(validate_features))
    actual_coords = validate_features[actual_start_idx:actual_end_idx, :2]

    # 可视化部分（与train.py风格一致）
    plt.figure(figsize=(18, 6))
    
    # 子图1: 原始数据与平滑数据对比（使用验证数据）
    plt.subplot(1, 3, 1)
    plt.plot(validate_features[:, 0], validate_features[:, 1], 'b-', alpha=0.5, label='原始数据')
    plt.plot(validate_smoothed[:, 0], validate_smoothed[:, 1], 'g-', label='平滑后数据')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('验证数据平滑前后对比')
    plt.legend()
    
    # 子图2: 验证集预测对比 - 散点图
    plt.subplot(1, 3, 2)
    plt.scatter(y_val_lonlat[:, 0], y_val_lonlat[:, 1], c='b', marker='o', alpha=0.6, label='实际位置')
    plt.scatter(y_val_pred_lonlat[:, 0], y_val_pred_lonlat[:, 1], c='r', marker='x', alpha=0.6, label='预测位置')
    
    # 添加连接线显示预测误差
    for i in range(len(y_val_lonlat)):
        plt.plot([y_val_lonlat[i, 0], y_val_pred_lonlat[i, 0]], 
                 [y_val_lonlat[i, 1], y_val_pred_lonlat[i, 1]], 
                 'gray', linestyle=':', alpha=0.3)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('验证集预测对比(散点图)')
    plt.legend()
    plt.grid(True)
    
    # 子图3: 验证数据轨迹与预测轨迹对比
    plt.subplot(1, 3, 3)
    plt.plot(validate_features[:, 0], validate_features[:, 1], 'b-', alpha=0.3, label='验证轨迹')
    # 绘制选中的n_steps个输入点
    plt.plot(validate_features[start_idx:start_idx+n_steps, 0], 
             validate_features[start_idx:start_idx+n_steps, 1], 
             'go-', markersize=8, label='输入序列')
    # 绘制实际轨迹（如果有足够的数据）
    if actual_end_idx > actual_start_idx:
        plt.plot(actual_coords[:, 0], actual_coords[:, 1], 
                 'bo-', markersize=8, label='实际轨迹')
    # 绘制预测轨迹
    plt.plot(predicted_coords[:, 0], predicted_coords[:, 1], 
             'rx--', markersize=8, linewidth=2, label='预测轨迹')
    # 标记起始点
    plt.scatter(validate_features[start_idx, 0], validate_features[start_idx, 1], 
               c='green', s=150, marker='*', label='起始点')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('验证数据轨迹与预测轨迹对比')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存可视化结果
    plt.savefig(os.path.join(save_dir, 'validation_results.png'))
    plt.show()

    return model, scaler_lon_lat, scaler_speed, scaler_course

if __name__ == "__main__":
    # 使用示例：
    # 1. 指定已训练模型和scalers的目录
    model_dir = "checkpoint/2025-07-23-15-05-12"  # 替换为实际的模型目录
    
    # 2. 指定新数据路径
    new_data_path = "./data/ais_data.csv"  # 替换为实际的新数据路径
    
    # 3. 执行微调
    finetune(model_dir, new_data_path)
