import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers, Model
import tensorflow as tf


# 1. 增强版数据预处理
def preprocess_data(data_path):
    data = pd.read_csv(data_path, parse_dates=['timestamp'])
    data = data.sort_values('timestamp')

    # 添加相对移动量特征
    data['dlon'] = data['longitude'].diff().fillna(0)
    data['dlat'] = data['latitude'].diff().fillna(0)

    features = data[['longitude', 'latitude', 'speed', 'course', 'dlon', 'dlat']]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)

    return scaler, scaled_data, data[['longitude', 'latitude']].values  # 保留原始坐标用于后续计算


# 2. 序列生成函数（支持多步输出）
def create_sequences(data, input_steps=60, output_steps=10):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i:i + input_steps])
        y.append(data[i + input_steps: i + input_steps + output_steps, :2])  # 未来10个点的经纬度
    return np.array(X), np.array(y)


# 3. 位置编码层
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

    def get_angles(self, position, i):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(self.d_model))
        return position * angles

    def call(self, inputs):
        seq_length = inputs.shape[1]
        position = np.arange(seq_length)[:, np.newaxis]
        div_term = self.get_angles(position, np.arange(self.d_model)[np.newaxis, :])

        # 正弦波编码
        pe = np.zeros((seq_length, self.d_model))
        pe[:, 0::2] = np.sin(div_term[:, 0::2])
        pe[:, 1::2] = np.cos(div_term[:, 1::2])

        return inputs + pe[np.newaxis, ...]


# 4. Transformer模型构建
def build_transformer_model(input_shape, output_steps, d_model=128):
    inputs = layers.Input(shape=input_shape)

    # 特征嵌入
    x = layers.Dense(d_model)(inputs)  # [batch, seq_len, d_model]

    # 位置编码
    x = PositionalEncoding(d_model)(x)

    # Transformer编码器
    for _ in range(2):
        attn_output = layers.MultiHeadAttention(
            num_heads=4, key_dim=d_model // 4)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        ffn_output = layers.Dense(d_model)(x)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # 解码部分（简化版）
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(output_steps * 2)(x)  # 输出10个时间步的经纬度
    outputs = layers.Reshape((output_steps, 2))(outputs)

    return Model(inputs, outputs)


# 5. 自定义评估指标（Haversine距离）
def haversine_distance(y_true, y_pred):
    """
    y_true: 归一化的真实坐标 (batch_size, output_steps, 2)
    y_pred: 归一化的预测坐标 (batch_size, output_steps, 2)
    """
    # 将归一化坐标转换为实际经纬度（假设经度归一化到[0,1]，纬度归一化到[0,1]）
    # 注意：需要根据实际数据范围调整以下转换参数
    lon_true = y_true[..., 0] * 360 - 180  # 假设经度范围[-180, 180]->[0,1]
    lat_true = y_true[..., 1] * 180 - 90  # 假设纬度范围[-90, 90]->[0,1]
    lon_pred = y_pred[..., 0] * 360 - 180
    lat_pred = y_pred[..., 1] * 180 - 90
    # 转换为弧度
    lon_true = tf.cast(lon_true, tf.float32) * (np.pi / 180.0)
    lat_true = tf.cast(lat_true, tf.float32) * (np.pi / 180.0)
    lon_pred = tf.cast(lon_pred, tf.float32) * (np.pi / 180.0)
    lat_pred = tf.cast(lat_pred, tf.float32) * (np.pi / 180.0)
    # 计算差值
    dlon = lon_pred - lon_true
    dlat = lat_pred - lat_true
    # Haversine公式
    a = tf.sin(dlat / 2.0) ** 2 + tf.cos(lat_true) * tf.cos(lat_pred) * tf.sin(dlon / 2.0) ** 2
    c = 2 * tf.math.atan2(tf.sqrt(a), tf.sqrt(1 - a))
    R = 6371.0  # 地球半径（公里）
    distance = R * c
    return tf.reduce_mean(distance)


# 主程序
if __name__ == "__main__":
    # 超参数设置
    INPUT_STEPS = 60  # 输入序列长度（60秒）
    OUTPUT_STEPS = 10  # 预测序列长度（10秒）
    BATCH_SIZE = 64
    EPOCHS = 100

    # 数据预处理
    scaler, processed_data, original_coords = preprocess_data('./data/wave_ship_data.csv')

    # 创建序列
    X, y = create_sequences(processed_data, INPUT_STEPS, OUTPUT_STEPS)

    # 划分数据集
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 构建模型
    model = build_transformer_model(
        input_shape=(INPUT_STEPS, X_train.shape[2]),
        output_steps=OUTPUT_STEPS
    )

    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=[haversine_distance]
    )

    # 训练模型
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # 预测与评估
    y_pred = model.predict(X_test)


    # 反归一化处理
    def inverse_transform(y_pred, scaler, X_last):
        """ 使用scaler的正确逆变换方法 """
        batch_size = y_pred.shape[0]
        output_steps = y_pred.shape[1]

        # 重构完整特征矩阵
        dummy = np.zeros((batch_size, output_steps, 6))
        dummy[:, :, :2] = y_pred  # 预测的经纬度

        # 修正维度问题：X_last已经是最后一个时间步的数据（shape: [batch_size, 4]）
        # 原始错误代码：last_features = X_last[:, -1, 2:] （三维索引）
        # 正确代码应该是：
        last_features = X_last[:, 2:]  # 直接取二维数据的后4个特征 [speed, course, dlon, dlat]

        # 将最后时刻的特征扩展到所有预测步
        dummy[:, :, 2:] = np.tile(last_features[:, np.newaxis, :], (1, output_steps, 1))

        # 逆变换
        dummy_2d = dummy.reshape(-1, 6)
        scaled_back = scaler.inverse_transform(dummy_2d)
        return scaled_back[:, :2].reshape(batch_size, output_steps, 2)


    # 我们只需要非经纬度特征（speed, course, dlon, dlat）即后4个特征
    # X_last_test_features = X_last_test[:, 2:]  # shape: [batch_size, 4]
    # 提取每个测试样本最后一个时间步的非经纬度特征
    X_last_test = X_test[:, -1, :]
    y_pred_geo = inverse_transform(y_pred, scaler, X_last_test)
    y_true_geo = inverse_transform(y_test, scaler, X_last_test)

    # 计算地理距离误差
    distances = []
    for i in range(len(y_test)):
        for t in range(OUTPUT_STEPS):
            d = haversine_distance(
                y_true_geo[i, t][np.newaxis, np.newaxis],
                y_pred_geo[i, t][np.newaxis, np.newaxis]
            )
            distances.append(d)
    print(f"平均地理误差：{np.mean(distances):.2f} 公里")

    # 可视化训练过程
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.legend()

    # 可视化轨迹预测
    plt.subplot(1, 2, 2)
    sample_id = np.random.randint(len(y_test))
    plt.plot(y_true_geo[sample_id, :, 0], y_true_geo[sample_id, :, 1],
             'go-', label='Actual Path', markersize=3)
    plt.plot(y_pred_geo[sample_id, :, 0], y_pred_geo[sample_id, :, 1],
             'rs--', label='Predicted Path', markersize=3)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Trajectory Prediction (Sample {sample_id})')
    plt.legend()
    plt.tight_layout()
    plt.show()
