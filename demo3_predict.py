import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt


class Encoder(keras.models.Model):
    def __init__(self, units=128):
        """ attent_size:自注意力机制的维度; units: Lstm 输出维度 """
        super().__init__()
        self.attention = keras.layers.Attention()
        self.lstm = keras.layers.LSTM(units, return_sequences=True, return_state=True
                                       # , activity_regularizer=tf.keras.regularizers.l2(0.05) 
                                        , kernel_regularizer=tf.keras.regularizers.l2(0.003)
                                      ) 
        self.lstm1 = keras.layers.LSTM(units, return_sequences=True, return_state=True
                                       # , activity_regularizer=tf.keras.regularizers.l2(0.05) 
                                        , kernel_regularizer=tf.keras.regularizers.l2(0.003)
                                      ) 
        self.dropout = keras.layers.Dropout(rate=0.3)

        
    def call(self, inputs):
        encoder_output, state_h1, state_c1 = self.lstm(inputs)
        # self.dropout(encoder_output)
        encoder_output, state_h2, state_c2 = self.lstm1(encoder_output)
        # print(encoder_output.shape)
        # print(dense.shape)
        return encoder_output, [state_h1, state_c1], [state_h2, state_c2] #[state_h, state_c]


class Decoder(keras.models.Model):
    def __init__(self, units = 128, label = 1, pre_k=None):
        """ units: Lstm 输出维度; label:最后输出的结果的维度 """
        super().__init__()
        self.pre_k = pre_k
        self.lstm = keras.layers.LSTMCell(units # , return_sequences=True, return_state=True
                                    , activity_regularizer=tf.keras.regularizers.l2(0.003)
                                    # , kernel_regularizer=tf.keras.regularizers.l2(0.05)
                                      )
        self.lstm1 = keras.layers.LSTMCell(units # , return_sequences=True, return_state=True 
                                        , activity_regularizer=tf.keras.regularizers.l2(0.003)
                                        # , kernel_regularizer=tf.keras.regularizers.l2(0.05)
                                      )

        self.attention = keras.layers.Attention()
        self.MultiHeadAttention = tf.keras.layers.MultiHeadAttention(num_heads=3, key_dim=2)
        self.dropout = keras.layers.Dropout(rate=0.3)
        self.dense = keras.layers.Dense(label) 
        self.dense1 = keras.layers.Dense(label) 
        self.dense2 = keras.layers.Dense(units)
        self.reshape = keras.layers.Reshape((1,-1))
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)


    def call(self, inputs, encoder_state1, encoder_state2, encoder_output):
        att_values, weights = self.MultiHeadAttention(inputs, inputs, return_attention_scores=True)
        att_values = self.layernorm(inputs + att_values)
        # att_values = tf.add(inputs, att_values)
        att_values = tf.reshape(att_values, shape=[-1, att_values.shape[-1] * att_values.shape[-2]])
        shape = int(inputs.shape[1] * inputs.shape[2])
        inputs = keras.layers.Reshape([shape])(inputs) 
                
        decoder_output2, state_h2 = self.lstm(inputs, states=encoder_state1)
        decoder_output, state_h = self.lstm1(decoder_output2, states=encoder_state2)
        decoder_output1 = self.reshape(decoder_output)
        attention_output, attention_scores1 = self.attention([decoder_output1, encoder_output], return_attention_scores=True)
                
        att_values = self.dense2(att_values)
        attention_output = tf.reshape(attention_output, shape=[-1, attention_output.shape[-1]])
        contanct_input = keras.layers.concatenate([attention_output, decoder_output, att_values], axis=1)
        print(contanct_input.shape,'+++')

        output = self.dense(contanct_input)
        output2 = self.dense1(contanct_input)
        print(output.shape)
 
        return output, output2, [weights, attention_scores1]


# 加载模型和标准化参数
# 加载模型时指定自定义层
model = keras.models.load_model(
    './checkpoint/trajectory_model.h5',
    custom_objects={'Encoder': Encoder, 'Decoder': Decoder}
)
scaler_params = joblib.load('./checkpoint/scaler_params.pkl')

def preprocess_data(raw_data):
    """预处理流程（需与训练时一致）"""
    # 1. 过滤纬度大于62的数据
    filtered_data = raw_data[raw_data["Latitude_N"] < 62].copy()
    # 2. 选择需要的特征列
    need_col = ['Longitude_W', 'Latitude_N', 'speed_knots','trueHeading']
    processed_data = filtered_data[need_col]
    return processed_data

def create_sequences(data, sequence_length=5):
    """创建时间序列数据"""
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i+sequence_length])
    return np.array(sequences)

# 加载新数据
new_data = pd.read_csv('./data/validate_data.csv')  # 修改为实际数据路径

# 数据预处理
processed_data = preprocess_data(new_data)

# 归一化
# 标准化处理（假设scaler_params包含所有特征的均值和标准差）
mean = np.array([
    scaler_params['long_mean'], 
    scaler_params['lat_mean'],
    scaler_params['speed_mean'], 
    scaler_params['heading_mean']
])
std = np.array([
    scaler_params['long_std'],
    scaler_params['lat_std'],
    scaler_params['speed_std'],
    scaler_params['heading_std']
])
scaled_data = (processed_data.values - mean) / std  # 标准化处理

# 创建时间序列
sequence_length = 5  # 与训练时的num参数一致
input_sequences = create_sequences(scaled_data, sequence_length)

# 预测
pred_long, pred_lat = model.predict(input_sequences)

# 逆标准化
pred_long_orig = pred_long * scaler_params['long_std'] + scaler_params['long_mean']
pred_lat_orig = pred_lat * scaler_params['lat_std'] + scaler_params['lat_mean']

# 获取真实数据（假设预测下一个时间步）
true_long = processed_data['Longitude_W'].values[sequence_length:]
true_lat = processed_data['Latitude_N'].values[sequence_length:]

# 可视化预测结果
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
# 绘制真实数据散点
plt.scatter(
    true_long, true_lat,
    c='blue', marker='o', 
    alpha=0.6, label='Actual Positions'
)
# 绘制预测数据散点
plt.scatter(
    pred_long_orig.flatten(), pred_lat_orig.flatten(),
    c='red', marker='x', 
    alpha=0.6, label='Predicted Positions'
)
# 添加图例和标签
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.title('Actual vs Predicted Trajectory Comparison', fontsize=14)
plt.legend()
plt.grid(True)
# 调整布局并保存
plt.tight_layout()
# plt.savefig('trajectory_comparison_scatter.png', dpi=300)


# 轨迹图
plt.subplot(1,2,2)

# 绘制真实轨迹（带半透明散点）
plt.scatter(
    true_long, true_lat,
    c='blue', s=15, 
    alpha=0.3, label='Actual Points'
)
plt.plot(
    true_long, true_lat,
    color='blue', linewidth=1.5,
    alpha=0.8, label='Actual Trajectory'
)

# 绘制预测轨迹（带半透明散点）
plt.scatter(
    pred_long_orig.flatten(), pred_lat_orig.flatten(),
    c='red', s=15, marker='x',
    alpha=0.3, label='Predicted Points'
)
plt.plot(
    pred_long_orig.flatten(), pred_lat_orig.flatten(),
    color='red', linestyle='--', linewidth=1.5,
    alpha=0.8, label='Predicted Trajectory'
)

# 添加图例和标签
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.title('Actual vs Predicted Trajectory Comparison', fontsize=14)
plt.legend(loc='upper left')  # 合并图例项
plt.grid(True, linestyle='--', alpha=0.5)

# 调整布局并保存
plt.tight_layout()
# plt.savefig('trajectory_comparison_line.png', dpi=300)
plt.show()



# 保存预测结果
# result_df = pd.DataFrame({
#     'Predicted_Longitude': pred_long_orig.flatten(),
#     'Predicted_Latitude': pred_lat_orig.flatten()
# })
# result_df.to_csv('predicted_coordinates.csv', index=False)
