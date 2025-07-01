# train.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Nadam
import joblib


def preprocess_data(data_path):
    data = pd.read_csv(data_path, parse_dates=['timestamp'])
    data = data.sort_values('timestamp')
    features = data[['longitude', 'latitude', 'speed', 'course']]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    return scaler, scaled_data


def create_sequences(data, n_steps=5):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps, :2])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(2))
    optimizer = Nadam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model


if __name__ == "__main__":
    # 参数设置
    n_steps = 5
    batch_size = 100
    epochs = 20

    # 创建保存目录
    os.makedirs('checkpoint', exist_ok=True)

    # 数据预处理
    scaler, processed_data = preprocess_data('./data/wave_ship_data.csv')

    # 创建数据集
    X, y = create_sequences(processed_data, n_steps)

    # 划分数据集
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 构建并训练模型
    model = build_lstm_model((n_steps, X_train.shape[2]))
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.2)

    # 保存模型和scaler
    model.save('checkpoint/model.h5')
    joblib.dump(scaler, 'checkpoint/scaler.pkl')
    print("训练完成，模型已保存至checkpoint目录")
