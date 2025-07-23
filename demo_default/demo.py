import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split  # 新增导入
from keras.models import Sequential, Model
from keras.optimizers import Nadam
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
import tensorflow as tf
from keras.layers import Input, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, concatenate, Reshape, Lambda, RepeatVector, BatchNormalization
from keras.regularizers import l2
from scipy.signal import savgol_filter
import mplcursors

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
def preprocess_data(data_path, smooth_window=5, smooth_method='savitzky_golay'):
    data = pd.read_csv(data_path, parse_dates=['timestamp'])
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
    
    return scaler_lon_lat, scaler_speed, scaler_course, scaled_data, features, smoothed_features, data  # 返回原始数据框用于后续处理

# 2. 构建监督学习序列（保持原样）
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

# 3. LSTM模型构建（保持原样）
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape, return_sequences=False))
    model.add(Dense(4))  # 输出四个特征
    optimizer = Nadam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def build_enhanced_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(4)
    ])
    optimizer = Nadam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

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

def build_enhanced_simple_model(input_shape, output_steps=4):
    inputs = Input(shape=input_shape)
    x = LSTM(256, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = LSTM(128)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(output_steps)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Nadam(0.0001), loss='mse')
    return model

def build_encoder_decoder_model(input_shape, units=128):
    encoder_inputs = Input(shape=input_shape)
    
    # 第一层LSTM
    encoder_lstm1 = LSTM(units, return_sequences=True, kernel_regularizer=l2(0.003))(encoder_inputs)
    encoder_lstm1 = Dropout(0.3)(encoder_lstm1)
    
    # 第二层LSTM并获取最终状态
    encoder_output, state_h, state_c = LSTM(units, return_sequences=False, return_state=True, 
                                           kernel_regularizer=l2(0.003))(encoder_lstm1)
    
    # 使用Lambda层替代tf.expand_dims
    decoder_input = Lambda(lambda x: tf.expand_dims(x, axis=1))(encoder_output)
    
    # 注意力机制
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(decoder_input, decoder_input)
    attention = LayerNormalization(epsilon=1e-6)(attention + decoder_input)
    
    decoder_output = Dense(64, activation='relu')(attention)
    decoder_output = Dense(4)(decoder_output)
    # 使用Lambda层替代tf.squeeze
    decoder_output = Lambda(lambda x: tf.squeeze(x, axis=1))(decoder_output)
    
    model = Model(encoder_inputs, decoder_output)
    optimizer = Nadam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def build_enhanced_encoder_decoder(input_shape, units=256):
    # 编码器
    encoder_inputs = Input(shape=input_shape)
    x = LSTM(units, return_sequences=True)(encoder_inputs)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    encoder_output, state_h, state_c = LSTM(units, return_sequences=False, return_state=True)(x)
    
    # 解码器
    decoder_input = RepeatVector(1)(encoder_output)  # 替代Lambda层
    x = LSTM(units, return_sequences=True)(decoder_input, initial_state=[state_h, state_c])
    x = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(4)(x)
    
    model = Model(encoder_inputs, outputs)
    optimizer = Nadam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# 4. 迭代预测函数（修改反归一化方式）
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

# 5. 主程序
if __name__ == "__main__":
    # 参数设置
    n_steps = 10
    batch_size = 32
    epochs = 500
    predict_steps = 5
    smooth_window = 7  # 平滑窗口大小
    smooth_method = 'savitzky_golay'  # 平滑方法: 'moving_average', 'exponential', 'savitzky_golay'
    random_state = 42  # 随机种子确保可重复性

    # 数据预处理（获取多个Scaler）
    scaler_lon_lat, scaler_speed, scaler_course, processed_data, raw_features, smoothed_features, data_df = preprocess_data(
        './data/ais_data.csv', 
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

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        verbose=1,
                        callbacks=callbacks)

    print("keys:",history.history.keys())
    # 常规预测与评估
    y_pred = model.predict(X_test).squeeze()
    
    # 反归一化处理（仅经纬度）
    y_pred_lonlat = scaler_lon_lat.inverse_transform(y_pred[:, :2]) 
    y_test_lonlat = scaler_lon_lat.inverse_transform(y_test[:, :2])
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_test_lonlat, y_pred_lonlat))
    print(f'Test RMSE: {rmse:.6f} degrees')
    
    # 迭代预测 - 加载新的validate_data.csv,选十个连续step作为输入，推理五个step，作图对比
    # 1. 加载validate_data.csv
    validate_data = pd.read_csv('./data/validate_data.csv', parse_dates=['timestamp'])
    validate_data = validate_data.sort_values('timestamp')
    validate_features = validate_data[['longitude', 'latitude', 'speed', 'course']].values
    # 对原始数据进行平滑处理
    validate_features = smooth_data(validate_features, window_size=smooth_window, method=smooth_method)
    # validate_features = validate_features[-50:]
    
    # 2. 归一化处理（使用之前训练数据的scaler）
    validate_lonlat = scaler_lon_lat.transform(validate_features[:, :2])
    validate_speed = scaler_speed.transform(validate_features[:, 2].reshape(-1, 1))
    validate_course = scaler_course.transform(validate_features[:, 3].reshape(-1, 1))
    validate_scaled = np.hstack((validate_lonlat, validate_speed, validate_course))
    
    # 3. 任选前35条数据中的10个连续时间步作为输入
    start_idx = np.random.randint(0, 25)  # 从0-25中随机选择起始点，确保有10个连续点
    initial_sequence = validate_scaled[start_idx:start_idx+n_steps]
    
    # 推理5个时间步的输出
    predicted_coords = iterative_predict(model, initial_sequence, scaler_lon_lat, predict_steps)
    
    # 获取对应的实际坐标（用于对比）
    actual_start_idx = start_idx + n_steps
    actual_end_idx = min(actual_start_idx + predict_steps, len(validate_features))
    actual_coords = validate_features[actual_start_idx:actual_end_idx, :2]

    
    # 可视化部分
    raw_coords = raw_features[:, :2]
    smoothed_coords = smoothed_features[:, :2]
    plt.figure(figsize=(18, 6))
    
    # 子图1: 原始数据与平滑数据对比（保持原样）
    plt.subplot(1, 3, 1)
    plt.plot(raw_coords[:, 0], raw_coords[:, 1], 'b-', alpha=0.5, label='原始数据')
    plt.plot(smoothed_coords[:, 0], smoothed_coords[:, 1], 'g-', label='平滑后数据')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('平滑前后数据对比')
    plt.legend()
    
    # 子图2: 测试集表现 - 修改为散点图对比
    plt.subplot(1, 3, 2)
    plt.scatter(y_test_lonlat[:, 0], y_test_lonlat[:, 1], c='b', marker='o', alpha=0.6, label='实际位置')
    plt.scatter(y_pred_lonlat[:, 0], y_pred_lonlat[:, 1], c='r', marker='x', alpha=0.6, label='预测位置')
    
    # 添加连接线显示预测误差
    for i in range(len(y_test_lonlat)):
        plt.plot([y_test_lonlat[i, 0], y_pred_lonlat[i, 0]], 
                 [y_test_lonlat[i, 1], y_pred_lonlat[i, 1]], 
                 'gray', linestyle=':', alpha=0.3)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('测试集预测对比(散点图)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(validate_features[:, 0], validate_features[:, 1], 'b-', alpha=0.3, label='验证轨迹')
    # 绘制选中的10个输入点
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
    plt.show()

