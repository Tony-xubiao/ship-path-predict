# predict.py
import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
from sklearn.metrics import mean_squared_error


def load_assets():
    model = load_model('checkpoint/model.h5',custom_objects={'mse': mean_squared_error} )
    scaler = joblib.load('checkpoint/scaler.pkl')
    return model, scaler


def prepare_input(data_path, scaler, n_steps=5):
    # 读取并预处理新数据
    data = pd.read_csv(data_path, parse_dates=['timestamp'])
    data = data.sort_values('timestamp')
    features = data[['longitude', 'latitude', 'speed', 'course']].values

    # 使用训练时的scaler进行归一化
    scaled_data = scaler.transform(features)

    # 构建最后一个有效序列
    sequence = scaled_data[-n_steps:]
    return np.array([sequence])


def inverse_scale(scaler, prediction):
    # 构建用于逆变换的dummy数组
    dummy = np.zeros((prediction.shape[0], scaler.n_features_in_))
    dummy[:, :2] = prediction
    return scaler.inverse_transform(dummy)[:, :2]


if __name__ == "__main__":
    # 加载模型和预处理工具
    model, scaler = load_assets()

    # 准备输入数据（假设有新数据文件）
    X_new = prepare_input('./data/validate_data.csv', scaler)

    # 进行预测
    prediction = model.predict(X_new)

    # 逆变换得到实际坐标
    actual_coords = inverse_scale(scaler, prediction)
    print(f"预测坐标（经度，纬度）: {actual_coords[0]}")
