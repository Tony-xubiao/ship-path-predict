import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, LSTM, Dense, RepeatVector, Dropout, LayerNormalization, MultiHeadAttention, Concatenate
from sklearn.base import BaseEstimator, TransformerMixin
from keras.optimizers import Nadam
import matplotlib.pyplot as plt
import tensorflow as tf

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False


# ====================== 1. 加载并预处理AIS数据 ======================
def load_ais_data(filepath, mmsi=None):
    """加载AIS数据，可选按MMSI筛选船舶"""
    df = pd.read_csv(filepath)
    if mmsi is not None:
        df = df[df['MMSI_'] == mmsi]  # 按MMSI筛选单船数据
    # 提取关键特征：经纬度、航向、速度
    data = df[['longitude', 'latitude', 'course', 'speed']].values
    timestamps = pd.to_datetime(df['timestamp']).values
    return data, timestamps

# 加载数据（替换为您的文件路径）
data, timestamps = load_ais_data('./data/wave_ship_data.csv', mmsi=None)  # 示例MMSI
print("原始数据形状:", data.shape)  # (n_samples, 4)

# 归一化（注意：经纬度需要单独缩放，避免失真）
scaler_lonlat = MinMaxScaler()  # 经纬度
scaler_others = MinMaxScaler()  # 航向、速度
data[:, :2] = scaler_lonlat.fit_transform(data[:, :2])
data[:, 2:] = scaler_others.fit_transform(data[:, 2:])

# ====================== 2. 生成多步训练数据 ======================
def create_ais_sequences(data, n_steps=5, predict_steps=3):
    """生成适用于AIS数据的多步序列"""
    X, y = [], []
    for i in range(len(data) - n_steps - predict_steps + 1):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps:i + n_steps + predict_steps])
    return np.array(X), np.array(y)

n_steps = 20       # 输入时间窗口（建议10-30步）
predict_steps = 5  # 预测未来5步
X, y = create_ais_sequences(data, n_steps, predict_steps)
print("输入数据形状:", X.shape)  # (samples, n_steps, 4)
print("标签数据形状:", y.shape)  # (samples, predict_steps, 4)

# ====================== 3. 构建AIS专用模型 ======================
def build_ais_model(input_shape, predict_steps, units=128):
    """针对AIS数据优化的Encoder-Decoder模型"""
    # Encoder
    encoder_inputs = Input(shape=input_shape)
    x = LSTM(units, return_sequences=True, activation='tanh')(encoder_inputs)
    x = Dropout(0.2)(x)
    encoder_output, state_h, state_c = LSTM(units, return_state=True)(x)
    
    # Decoder
    decoder_input = RepeatVector(predict_steps)(encoder_output)
    x = LSTM(units, return_sequences=True)(decoder_input, initial_state=[state_h, state_c])
    
    # Attention + Skip Connection
    attn = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = LayerNormalization()(x + attn)
    
    # 输出层（经纬度用tanh激活，其他用线性）
    lonlat = Dense(2, activation='tanh')(x)  # 经纬度在-1到1之间
    others = Dense(2)(x)                     # 航向和速度
    outputs = Concatenate(axis=-1)([lonlat, others])  # 使用Keras的Concatenate层
    
    model = Model(encoder_inputs, outputs)
    model.compile(optimizer=Nadam(0.001), loss='mse')
    return model

model = build_ais_model(input_shape=(n_steps, 4), predict_steps=predict_steps)
model.summary()

# ====================== 4. 训练模型 ======================
history = model.fit(
    X, y,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
)

# 绘制训练曲线
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()

# ====================== 5. 多步预测与可视化 ======================
def predict_ais_trajectory(model, initial_seq, scaler_lonlat, scaler_others):
    """预测未来轨迹并反归一化"""
    pred = model.predict(initial_seq[np.newaxis, ...])[0]  # (predict_steps, 4)
    pred_lonlat = scaler_lonlat.inverse_transform(pred[:, :2])
    pred_others = scaler_others.inverse_transform(pred[:, 2:])
    return pred_lonlat, pred_others

# 测试预测（取最后n_steps步作为输入）
test_seq = data[-n_steps:]
pred_lonlat, pred_others = predict_ais_trajectory(model, test_seq, scaler_lonlat, scaler_others)

print("\n预测结果（未来5步）:")
print("经度\t纬度\t航向\t速度")
for i in range(predict_steps):
    print(f"{pred_lonlat[i, 0]:.5f}\t{pred_lonlat[i, 1]:.5f}\t{pred_others[i, 0]:.1f}\t{pred_others[i, 1]:.1f}")

# 可视化轨迹对比
true_lonlat = scaler_lonlat.inverse_transform(data[-n_steps-predict_steps:, :2])
pred_full = np.vstack([true_lonlat[:n_steps], pred_lonlat])

plt.subplot(1,2,2)
plt.plot(true_lonlat[:, 0], true_lonlat[:, 1], 'bo-', label='真实轨迹')
plt.plot(pred_full[:, 0], pred_full[:, 1], 'ro--', label='预测轨迹')
plt.scatter(true_lonlat[n_steps-1, 0], true_lonlat[n_steps-1, 1], c='green', s=100, label='当前位置')
plt.legend()
plt.xlabel("经度")
plt.ylabel("纬度")
plt.title("AIS轨迹多步预测")
plt.grid()
plt.show()
