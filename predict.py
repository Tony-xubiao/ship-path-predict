# predict.py
import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
import os

from sklearn.metrics import mean_squared_error


def load_assets():
    """加载模型和预处理工具"""
    model = load_model('checkpoint/model_60-10.h5',custom_objects={'mse': mean_squared_error})

    # 加载独立特征Scaler
    scalers = {
        'longitude': joblib.load('checkpoint/scaler_longitude.pkl'),
        'latitude': joblib.load('checkpoint/scaler_latitude.pkl'),
        'speed': joblib.load('checkpoint/scaler_speed.pkl'),
        'course': joblib.load('checkpoint/scaler_course.pkl')
    }

    # 验证所有scaler存在
    missing = [k for k, v in scalers.items() if v is None]
    if missing:
        raise ValueError(f"缺少scaler文件: {', '.join(missing)}")

    return model, scalers


def prepare_input(data_path, scalers, n_steps=60):
    """准备输入数据（适配独立Scaler）"""
    # 读取并预处理新数据
    data = pd.read_csv(data_path, parse_dates=['timestamp'])
    data = data.sort_values('timestamp')
    features = data[['longitude', 'latitude', 'speed', 'course']].copy()

    # 检查数据长度是否足够
    if len(features) < n_steps:
        raise ValueError(f"至少需要{n_steps}条历史数据，当前只有{len(features)}条")

    # 对各特征独立归一化
    scaled_features = []
    for col in ['longitude', 'latitude', 'speed', 'course']:
        scaler = scalers[col]
        scaled_col = scaler.transform(features[[col]].values)
        scaled_features.append(scaled_col.ravel())

    # 合并特征并提取序列
    scaled_data = np.column_stack(scaled_features)
    sequence = scaled_data[-n_steps:]
    return np.array([sequence])  # 添加批次维度


def inverse_scale(scalers, predictions):
    """逆变换预测结果（仅需经纬度Scaler）"""
    # predictions形状：(num_samples, 10, 2)
    lon_scaler = scalers['longitude']
    lat_scaler = scalers['latitude']

    # 处理经度
    lon = predictions[..., 0].reshape(-1, 1)
    inverted_lon = lon_scaler.inverse_transform(lon)

    # 处理纬度
    lat = predictions[..., 1].reshape(-1, 1)
    inverted_lat = lat_scaler.inverse_transform(lat)

    # 重组为(10, 2)形状
    return np.column_stack([inverted_lon, inverted_lat]).reshape(predictions.shape)


if __name__ == "__main__":
    try:
        # 加载模型资源
        model, scalers = load_assets()

        # 准备输入数据（形状：1×60×4）
        X_new = prepare_input('./data/validate_data.csv', scalers)

        # 进行预测（输出形状：1×10×2）
        predicted_coords = model.predict(X_new)

        # 逆变换得到实际坐标
        actual_coords = inverse_scale(scalers, predicted_coords)[0]  # 取第一个批次

        # 格式化输出预测结果
        print("未来10秒预测轨迹：")
        for i, (lon, lat) in enumerate(actual_coords, 1):
            print(f"第{i}秒 -> 经度：{lon:.6f}, 纬度：{lat:.6f}")

    except Exception as e:
        print(f"预测失败：{str(e)}")
