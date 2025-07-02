使用说明：
准备AIS数据文件（csv格式），包含以下字段：

timestamp（时间戳）
longitude（经度）
latitude（纬度）
speed（速度）
course（航向）

# demo.py 使用过去的'longitude', 'latitude', 'speed', 'course' 数据，预测'longitude', 'latitude'两个参数，“过去”的长度取决于参数n_steps，只能预测下一秒数据

# demo2.py 使用过去的'longitude', 'latitude', 'speed', 'course' 数据，预测'longitude', 'latitude' ,'speed', 'course'四个参数，“过去”的长度取决于参数n_steps，可以迭代预测无限时长

运行demo2.py
包括训练、推理、可视化展示


可根据需要调整的超参数：

n_steps = 5       # 时间窗口大小
batch_size = 100  # 批处理量
epochs = 20       # 训练轮数
predict_steps = 30 # 迭代预测的时间长度
