import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Nadam
from sklearn.metrics import mean_squared_error


# 1. 数据预处理（修改y包含四个特征）
def preprocess_data(data_path):
    data = pd.read_csv(data_path, parse_dates=['timestamp'])
    data = data.sort_values('timestamp')
    features = data[['longitude', 'latitude', 'speed', 'course']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    return scaler, scaled_data


# 2. 构建监督学习序列（修改y包含四个特征）
def create_sequences(data, n_steps=5):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps, :4])  # 修改为预测四个特征
    return np.array(X), np.array(y)


# 3. LSTM模型构建（修改输出维度为4）
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(4))  # 输出四个特征
    optimizer = Nadam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')
    return model


# 新增迭代预测函数
def iterative_predict(model, initial_seq, scaler, predict_steps=30):
    current_seq = initial_seq.copy()
    predicted = []

    for _ in range(predict_steps):
        # 预测下一个时间步
        pred = model.predict(current_seq.reshape(1, *current_seq.shape))[0]
        # 保存归一化后的预测结果
        predicted.append(pred)
        # 更新序列：移除第一个时间步，添加预测结果到末尾
        current_seq = np.concatenate([current_seq[1:], pred.reshape(1, -1)])

    # 反归一化所有预测结果
    predicted = np.array(predicted)
    predicted_actual = scaler.inverse_transform(predicted)
    return predicted_actual[:, :2]  # 返回经纬度


# 4. 主程序（添加迭代预测功能）
if __name__ == "__main__":
    # 参数设置
    n_steps = 50
    batch_size = 20
    epochs = 2000
    predict_steps = 30  # 默认预测30个时间步

    # 数据预处理
    scaler, processed_data = preprocess_data('./data/wave_ship_data.csv')

    # 创建序列数据集
    X, y = create_sequences(processed_data, n_steps)

    # 划分训练集和测试集
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 构建并训练模型
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        verbose=1)

    # 常规预测与评估
    y_pred = model.predict(X_test)

    # 反归一化（直接使用完整四个特征进行转换）
    y_pred_actual = scaler.inverse_transform(y_pred)[:, :2]
    y_test_actual = scaler.inverse_transform(y_test)[:, :2]

    # 计算RMSE（仅经纬度）
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    print(f'Test RMSE: {rmse:.6f} degrees')

    # 迭代预测（使用测试集第一个样本作为初始序列）
    initial_sequence = X_test[0]
    predicted_coords = iterative_predict(model, initial_sequence, scaler, predict_steps)

    # 可视化迭代预测结果
    import matplotlib.pyplot as plt

    # 获取实际轨迹（测试集后续predict_steps个点）
    actual_steps = min(predict_steps, len(y_test))
    actual_coords = scaler.inverse_transform(y_test[:actual_steps])[:, :2]

    plt.figure(figsize=(12, 6))
    plt.plot(actual_coords[:, 0], actual_coords[:, 1], 'b-', label='Actual Track')
    plt.plot(predicted_coords[:, 0], predicted_coords[:, 1], 'r--',
             label=f'Predicted {predict_steps}steps')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Iterative Trajectory Prediction ({predict_steps} steps)')
    plt.legend()
    plt.show()
