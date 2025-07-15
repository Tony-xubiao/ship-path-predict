# predict.py
import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False

def load_assets(model_path):
    """加载模型和预处理工具"""
    model = load_model(f'{model_path}/model_finetuned_2025-07-15-10-29-55_to_2025-07-15-10-53-55.h5',
                       custom_objects={'mse': mean_squared_error})

    # 加载独立特征Scaler
    scalers = {
        'longitude': joblib.load(f'{model_path}/scaler_finetuned_2025-07-15-10-29-55_longitude.pkl'),
        'latitude': joblib.load(f'{model_path}/scaler_finetuned_2025-07-15-10-29-55_latitude.pkl'),
        'speed': joblib.load(f'{model_path}/scaler_finetuned_2025-07-15-10-29-55_speed.pkl'),
        'course': joblib.load(f'{model_path}/scaler_finetuned_2025-07-15-10-29-55_course.pkl')
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
    return np.array([sequence]), data  # 添加批次维度并返回原始数据


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


def plot_trajectory(original_data, predicted_coords):
    """绘制原始轨迹和预测轨迹"""
    plt.figure(figsize=(12, 8))

    # 绘制原始轨迹
    plt.plot(original_data['longitude'], original_data['latitude'],
             'b-', label='原始轨迹', alpha=0.7)

    # 绘制最后10个原始点
    last_10_original = original_data[['longitude', 'latitude']].values[-10:]
    plt.plot(last_10_original[:, 0], last_10_original[:, 1],
             'bo', markersize=6, label='最后10个原始点')

    # 绘制预测轨迹
    plt.plot(predicted_coords[:, 0], predicted_coords[:, 1],
             'r--', label='预测轨迹', linewidth=2)
    plt.plot(predicted_coords[:, 0], predicted_coords[:, 1],
             'ro', markersize=6, label='预测点')

    # 连接最后原始点和第一个预测点
    last_point = original_data[['longitude', 'latitude']].values[-1]
    first_pred = predicted_coords[0]
    plt.plot([last_point[0], first_pred[0]], [last_point[1], first_pred[1]],
             'g--', linewidth=1, alpha=0.5)

    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.title('船舶轨迹预测')
    plt.legend()
    plt.grid(True)

    # 调整坐标轴比例，使1度经度和1度纬度在图上显示长度相同
    plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        # 加载模型资源
        model_path = "checkpoint/2025-07-15-10-53-55"
        model, scalers = load_assets(model_path)

        # 准备输入数据（形状：1×60×4）
        X_new, original_data = prepare_input('./data/validate_data_copy.csv', scalers)

        # 进行预测（输出形状：1×10×2）
        predicted_coords = model.predict(X_new)

        # 逆变换得到实际坐标
        actual_coords = inverse_scale(scalers, predicted_coords)[0]  # 取第一个批次

        # 格式化输出预测结果
        print("未来10秒预测轨迹：")
        for i, (lon, lat) in enumerate(actual_coords, 1):
            print(f"第{i}秒 -> 经度：{lon:.6f}, 纬度：{lat:.6f}")

        # 绘制轨迹图
        plot_trajectory(original_data, actual_coords)

    except Exception as e:
        print(f"预测失败：{str(e)}")
