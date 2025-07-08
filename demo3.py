
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from math import sqrt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False

trej1 = pd.read_csv(r"./data/real_ais_data.csv")
# data_per1 = data_per1.iloc[150:,:]
data_per1 = trej1.copy()
# data_per1['Time'] = data_per['Time']
data_per1.drop(index=data_per1[data_per1["Latitude_N"] >= 62].index.values, axis=0, inplace=True)
data_orig = data_per1.copy()

def Standardization(data):
    return (data - data.mean(axis=0)) / data.std(axis=0)

def RMSE(y_pre, y_real):
    return sqrt(np.sum((y_pre - y_real) ** 2) / y_pre.shape[0])

def loss_function(predicted_x , target ):
    loss = np.sum(np.square(predicted_x - target) , axis= 1)/(predicted_x.size()[1])
    loss = np.sum(loss)/loss.shape[0]
    return loss
def samples(data, num):
    dfs = []
    for k in range(0, (data.shape[0] - num) + 1):
        dfs.append(data[k : k + num])
    seq_data = pd.concat(dfs, ignore_index=True)
    return seq_data
# 去除不需要的列
def data_columns(data, need_col):
    index = list(data.columns)
    drop_col = list(set(index) - set(need_col))
    data.drop(drop_col, inplace=True, axis=1)
    data.drop(index=data[data["Latitude_N"] >= 62].index.values, axis=0, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def predict_y_samples(data, pre_k, num):
    data = Standardization(data)
    for i in range(pre_k):
        data.loc[data.iloc[(num-1):-(i+1), :].index, 'per_long_'+str(i+1)] = data.iloc[(num+i):,:].Longitude_W.values
        data.loc[data.iloc[(num-1):-(i+1), :].index, 'per_lat_'+str(i+1)] = data.iloc[(num+i):,:].Latitude_N.values
        
    data.drop(index=data.iloc[data.shape[0]-pre_k:, :].index.values, axis=0, inplace = True)
    return data

def random_point_get_samples(data, num, probability, need_col, pre_k, input_size):
    index = data.index[data.index >= num]
    train_index = random.sample(list(index), int(index.shape[0] * probability))
    test_index = list(set(index) - set(train_index))
    
    # 使用列表收集各部分DataFrame
    train_x_parts = []
    test_x_parts = []
    train_y = {j+1: [] for j in range(pre_k)}
    test_y = {j+1: [] for j in range(pre_k)}
    
    for i in train_index:
        train_x_parts.append(data.loc[int(i-(num-1)):int(i), need_col])
        for j in range(pre_k):
            train_y[j+1].append(data.loc[int(i), ['per_long_'+str(j+1), 'per_lat_'+str(j+1)]].values)
    
    for i in test_index:
        test_x_parts.append(data.loc[int(i-(num-1)):int(i), need_col])
        for j in range(pre_k):
            test_y[j+1].append(data.loc[int(i), ['per_long_'+str(j+1), 'per_lat_'+str(j+1)]].values)
    
    # 使用concat合并DataFrame
    train_x = pd.concat(train_x_parts, ignore_index=True)
    test_x = pd.concat(test_x_parts, ignore_index=True)
    
    return train_index, train_x.values.reshape(-1, num, input_size), test_x.values.reshape(-1, num, input_size), train_y, test_y


def yshape(train_y, test_y):
    ytrain, ytest = {}, {}
    for i in list(train_y.keys()):
        ytrain[i] = np.array(train_y[i])
        ytest[i] = np.array(test_y[i])
        if i <= 1:
            y_train = ytrain[i]
            y_test = ytest[i]
        else:
            y_train = np.c_[y_train, ytrain[i]]
            y_test = np.c_[y_test, ytest[i]]            
    return y_train.reshape(-1,y_train.shape[-1]), y_test.reshape(-1, y_test.shape[-1])


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

num = 10
time_step = num
pre_k = 1
epochs = 100
batch_size = 512
need_col = ['Longitude_W', 'Latitude_N', 'speed_knots','trueHeading'] 
data_per1 = data_columns(data_per1, need_col=need_col)
input_size = int(len(need_col))
data_per1 = predict_y_samples(data_per1, pre_k=pre_k, num=num)
datz= data_per1.copy()
train_index, train_x, test_x, train_y, test_y = random_point_get_samples(data_per1, input_size=input_size
                                                                         , num=num
                                                                         , probability=0.6
                                                                         , need_col=need_col, pre_k=pre_k)
y_train, y_test = yshape(train_y, test_y)
            
encoder_input = keras.layers.Input(shape=([time_step, train_x.shape[2]]))
encoder_output,encoder_state1, encoder_state2=Encoder()(encoder_input)
            
output_long, output_lat, weights = Decoder(pre_k=pre_k)(encoder_input, encoder_state1, encoder_state2, encoder_output) #_long, output_lat

model = keras.models.Model([encoder_input], [output_long, output_lat])
# model.summary()
model.compile(loss = keras.losses.MeanSquaredError()
                          ,optimizer = keras.optimizers.Adam(learning_rate=0.01) 
                          ,metrics=keras.metrics.RootMeanSquaredError()) #MeanAbsoluteError()
history = model.fit([train_x], [y_train[:,0], y_train[:,1]], epochs=epochs
                                , batch_size = batch_size
                                , validation_data=([test_x], [y_test[:,0], y_test[:,1]]))
model.save('./checkpoint/trajectory_model.h5')  # 保存完整模型
# 获取原始数据的均值和标准差（在数据标准化前的原始数据）
orig_mean_long = data_orig['Longitude_W'].mean()
orig_std_long = data_orig['Longitude_W'].std()
orig_mean_lat = data_orig['Latitude_N'].mean()
orig_std_lat = data_orig['Latitude_N'].std()
orig_mean_speed = data_orig['speed_knots'].mean()
orig_std_speed = data_orig['speed_knots'].std()
orig_mean_head = data_orig['trueHeading'].mean()
orig_std_head = data_orig['trueHeading'].std()
# 保存标准化参数
import joblib
scaler_params = {
    'long_mean': orig_mean_long,
    'long_std': orig_std_long,
    'lat_mean': orig_mean_lat,
    'lat_std': orig_std_lat,
    'speed_std': orig_std_speed,
    'speed_mean': orig_mean_speed,
    'heading_mean': orig_mean_head,
    'heading_std': orig_std_head
}
joblib.dump(scaler_params, './checkpoint/scaler_params.pkl')

score_ourmodel = model.evaluate([test_x], [y_test[:,0], y_test[:,1]])


plt.figure(figsize=(18, 6))
# plt.subplot(2, 3, 1)
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.xlabel('Epochs %s batch_size %s' % (epochs, batch_size), fontsize = 12)
# plt.ylabel('MSE', fontsize = 12)
# # plt.show()
# plt.subplot(2, 3, 2)
# plt.plot(history.history[list(history.history.keys())[-6]], label='train')
# plt.plot(history.history[list(history.history.keys())[-1]], label='test')
# plt.legend()
# plt.xlabel('Epochs %s batch_size %s' % (epochs, batch_size), fontsize = 12)
# plt.ylabel('RMSE', fontsize = 12)
# # plt.show()
# plt.subplot(2, 3, 3)
# plt.plot(history.history[list(history.history.keys())[-7]], label='train')
# plt.plot(history.history[list(history.history.keys())[-2]], label='test')
# plt.legend()
# plt.xlabel('Epochs %s batch_size %s' % (epochs, batch_size), fontsize = 12)
# plt.ylabel('RMSE', fontsize = 12)
# # plt.show()
# print("keys:",history.history.keys())

plt.subplot(1, 3, 1)
plt.plot(history.history["decoder_1_root_mean_squared_error"], label='Training Loss (RMSE)')
plt.plot(history.history["val_decoder_1_root_mean_squared_error"], label='Validation Loss (RMSE)')
plt.title('Training and Validation Loss Curve', fontsize=10)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Mean Squared Error (RMSE)', fontsize=12)
plt.legend()
plt.grid(True)
# plt.tight_layout()

plt.subplot(1, 3, 2)
# 预测测试集
y_pred_long, y_pred_lat = model.predict(test_x)
# 逆标准化
y_test_long_orig = y_test[:,0] * orig_std_long + orig_mean_long
y_test_lat_orig = y_test[:,1] * orig_std_lat + orig_mean_lat
y_pred_long_orig = y_pred_long * orig_std_long + orig_mean_long
y_pred_lat_orig = y_pred_lat * orig_std_lat + orig_mean_lat
plt.plot(y_test_long_orig, y_test_lat_orig, 'b-', label='True Trajectory')
plt.plot(y_pred_long_orig, y_pred_lat_orig, 'r--', label='Predicted Trajectory')
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.title('测试集预测轨迹对比', fontsize=10)
plt.legend()
plt.grid(True)
# plt.savefig('trajectory_comparison.png')
# plt.show()

plt.subplot(1, 3, 3)
# 绘制散点对比图
plt.scatter(y_test_long_orig, y_test_lat_orig, c='b', alpha=0.6, label='True')
plt.scatter(y_pred_long_orig, y_pred_lat_orig, c='r', alpha=0.6, label='Predicted')
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.title('测试集预测散点图对比', fontsize=10)
plt.legend()
plt.grid(True)
# plt.savefig('scatter_comparison.png')
plt.show()