使用说明：
准备AIS数据文件（csv格式），包含以下字段：

timestamp（时间戳）
longitude（经度）
latitude（纬度）
speed（速度）
course（航向）

可根据需要调整的超参数：

n_steps = 5       # 时间窗口大小
batch_size = 100  # 批处理量
epochs = 20       # 训练轮数
