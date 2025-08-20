import sys
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shipai.model.model_load as ml

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False

def load_model_scaler(mmsi, model_code):
    """从指定目录加载最新的模型和scaler"""
    # 加载模型
    model = ml.load_model(mmsi, model_code)
    
    # 加载scaler
    scaler_lon_lat = ml.load_joblib(f"{ml.get_path(mmsi, model_code)}/scaler_lon_lat.pkl", 'scaler_lon_lat.pkl')
    scaler_speed = ml.load_joblib(f"{ml.get_path(mmsi, model_code)}/scaler_speed.pkl", 'scaler_speed.pkl')
    scaler_course = ml.load_joblib(f"{ml.get_path(mmsi, model_code)}/scaler_course.pkl", 'scaler_course.pkl')

    return model, scaler_lon_lat, scaler_speed, scaler_course

# 迭代预测函数
def iterative_predict(model, initial_seq, scaler_lon_lat, predict_steps):
    current_seq = initial_seq.copy()
    predicted = []
    
    for _ in range(predict_steps):
        pred = model.predict(current_seq[np.newaxis, ...])  # 保持三维输入
        pred = pred.reshape(-1)  # 展平为一维数组
        predicted.append(pred)
        # 更新序列（去掉第一个时间步，添加预测结果）
        current_seq = np.vstack([current_seq[1:], pred])
    
    predicted = np.array(predicted)
    predicted_lonlat = scaler_lon_lat.inverse_transform(predicted[:, :2])
    return predicted_lonlat

def preprocess_validation_data(sftp, data_path, scaler_lon_lat, scaler_speed, scaler_course):
    """预处理验证数据（不进行平滑处理）"""
    try:
        with sftp.open(data_path, 'r') as f:
            content = f.read().decode('utf-8')
            data = pd.read_csv(StringIO(content), parse_dates=['timestamp'])
    except Exception as e:
        print(f"读取 CSV 数据出错，请检查数据，错误原因: {e}")
        sys.exit(1)

    data = data.sort_values('timestamp')
    features = data[['longitude', 'latitude', 'speed', 'course']].values

    # 直接归一化（不进行平滑）
    lonlat_scaled = scaler_lon_lat.transform(features[:, :2])
    speed_scaled = scaler_speed.transform(features[:, 2].reshape(-1, 1))
    course_scaled = scaler_course.transform(features[:, 3].reshape(-1, 1))
    
    scaled_data = np.hstack((lonlat_scaled, speed_scaled, course_scaled))
    return scaled_data, features