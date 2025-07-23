import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# import matplotlib.pyplot as plt

# 生成模拟数据参数
np.random.seed(42)
total_points = 600  # 36000秒=10小时，每60s一条数据，10小时共600条
center_lon = 128.5  # 初始经度
center_lat = 30.0  # 初始纬度

# 生成时间序列（每秒一个时间戳）
start_time = datetime(2023, 1, 1, 0, 0, 0)
timestamps = [start_time + timedelta(seconds=i) for i in range(total_points)]

# 生成基本运动轨迹（东北方向）
t = np.linspace(0, 10 * np.pi, total_points)  # 10个波动周期

# 经度线性递增（稳定东移）
longitudes = center_lon + np.linspace(0, 5, total_points)  # 10小时东移5度

# 纬度添加正弦波动（北移基线+横向波动）
latitudes = center_lat + 0.1 * np.sin(t) + np.linspace(0, 2, total_points)

# 速度生成（10±2节）
base_speed = 10
speed = base_speed + np.random.normal(0, 0.3, total_points)

# 航向生成（基于运动方向）
dx = np.gradient(longitudes)
dy = np.gradient(latitudes)
course = np.degrees(np.arctan2(dy, dx)) % 360

# 添加合理噪声
# longitudes += np.random.normal(0, 2e-5, total_points)
# latitudes += np.random.normal(0, 2e-5, total_points)
# speed = np.clip(speed, 8, 12)

# 创建DataFrame
df = pd.DataFrame({
    "timestamp": timestamps,
    "longitude": longitudes,
    "latitude": latitudes,
    "speed": speed,
    "course": course
})

# 格式化精度
df["longitude"] = df["longitude"].round(6)
df["latitude"] = df["latitude"].round(6)
df["speed"] = df["speed"].round(1)
df["course"] = df["course"].round(1)

# 保存到CSV
df.to_csv("wave_ship_data.csv", index=False)
print("数据已保存到 wave_ship_data.csv")





