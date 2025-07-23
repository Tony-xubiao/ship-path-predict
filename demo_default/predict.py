import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib
import os
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False

def load_latest_model(checkpoint_dir):
    """从指定目录加载最新的模型和scaler"""
    model_path = os.path.join(checkpoint_dir, 'model.h5')
    
    # 加载模型
    model = load_model(model_path,
                       custom_objects={'mse': mean_squared_error})
    
    # 加载scaler
    scaler_lon_lat = joblib.load(os.path.join(checkpoint_dir, 'scaler_lon_lat.pkl'))
    scaler_speed = joblib.load(os.path.join(checkpoint_dir, 'scaler_speed.pkl'))
    scaler_course = joblib.load(os.path.join(checkpoint_dir, 'scaler_course.pkl'))
    
    return model, scaler_lon_lat, scaler_speed, scaler_course

def iterative_predict(model, initial_seq, scaler_lon_lat, predict_steps=30):
    """迭代预测函数"""
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

def single_step_predict(model, sequence):
    """单步预测函数"""
    pred = model.predict(sequence[np.newaxis, ...])
    return pred[0]  # 返回单个预测结果

def preprocess_validation_data(data_path, scaler_lon_lat, scaler_speed, scaler_course):
    """预处理验证数据（不进行平滑处理）"""
    data = pd.read_csv(data_path, parse_dates=['timestamp'])
    data = data.sort_values('timestamp')
    features = data[['longitude', 'latitude', 'speed', 'course']].values
    
    # 直接归一化（不进行平滑）
    lonlat_scaled = scaler_lon_lat.transform(features[:, :2])
    speed_scaled = scaler_speed.transform(features[:, 2].reshape(-1, 1))
    course_scaled = scaler_course.transform(features[:, 3].reshape(-1, 1))
    
    scaled_data = np.hstack((lonlat_scaled, speed_scaled, course_scaled))
    return scaled_data, features

def plot_comparison(input_start, actual, iterative_pred, input_seq):
    """绘制对比图"""
    plt.figure(figsize=(15, 6))
    # 迭代预测对比
    plt.plot(actual[:, 0], actual[:, 1], 'bo-', label='实际轨迹')
    plt.plot(iterative_pred[:, 0], iterative_pred[:, 1], 'mx--', label='迭代预测')
    plt.plot(input_seq[:, 0], input_seq[:, 1], 'r-', linewidth=2, alpha=0.8, label='输入序列')
    
    for i in range(predict_steps):
        plt.plot([actual[input_start+n_steps+i, 0], iterative_pred[i, 0]], 
                 [actual[input_start+n_steps+i, 1], iterative_pred[i, 1]], 
                 'gray', linestyle=':', alpha=0.3)
    
    
    plt.title('迭代预测对比')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 参数设置
    checkpoint_dir = "D:\code\PycharmProject\path-predict\demo_default\checkpoint/2025-07-23-15-05-12"
    validate_data_path = "D:\code\PycharmProject\path-predict\demo_default\data/validate_data.csv"
    n_steps = 20  # 需要与训练时的n_steps一致，不能随意修改
    predict_steps = 5  # 预测步数，随意修改
    input_start = 40 # 从验证集的数据的何处开始取n_steps个时间步长度的数据输入（起始点）,要小于等于len(数据集)-n_steps-predict_steps
    
    # 加载最新模型和scaler
    model, scaler_lon_lat, scaler_speed, scaler_course = load_latest_model(checkpoint_dir)
    
    # 预处理验证数据
    scaled_data, raw_features = preprocess_validation_data(
        validate_data_path, 
        scaler_lon_lat,
        scaler_speed,
        scaler_course
    )
    assert len(scaled_data) >= input_start + n_steps + predict_steps, "输入序列长度不足，请减小input_start的值或者增加数据量，需满足数据量>=input_start + n_steps + predict_steps"
    
    # 创建输入序列（取从input_start处开始的往后n_steps个时间步，长度固定n_steps）
    initial_sequence = scaled_data[input_start: input_start+n_steps]  # 使用最开始的连续序列
    
    # 进行迭代预测
    iterative_pred_lonlat = iterative_predict(model, initial_sequence, scaler_lon_lat, predict_steps)
    
    # 真实坐标(整个数据集)
    actual_coords = raw_features[:, :2]
    
    # 输入序列的真实坐标（长度固定n_steps）
    input_coords = raw_features[input_start : input_start+n_steps, :2]
    
    # 6. 可视化对比
    plot_comparison(
        input_start = input_start,
        actual=actual_coords,
        iterative_pred=iterative_pred_lonlat,
        input_seq=input_coords
    )
