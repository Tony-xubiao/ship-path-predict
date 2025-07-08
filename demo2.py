import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Nadam
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
import keras
import tensorflow as tf

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据预处理（使用多个Scaler）
def preprocess_data(data_path):
    data = pd.read_csv(data_path, parse_dates=['timestamp'])
    data = data.sort_values('timestamp')
    features = data[['longitude', 'latitude', 'speed', 'course']]
    
    # 初始化不同特征的Scaler
    scaler_lon_lat = MinMaxScaler()  # 经纬度共用
    scaler_speed = MinMaxScaler()    # 速度单独
    scaler_course = MinMaxScaler()   # 航向单独
    
    # 分别进行归一化
    lon_lat_scaled = scaler_lon_lat.fit_transform(features[['longitude', 'latitude']])
    speed_scaled = scaler_speed.fit_transform(features[['speed']].values.reshape(-1, 1))
    course_scaled = scaler_course.fit_transform(features[['course']].values.reshape(-1, 1))
    
    # 合并归一化后的数据
    scaled_data = np.hstack((lon_lat_scaled, speed_scaled, course_scaled))
    
    return scaler_lon_lat, scaler_speed, scaler_course, scaled_data

# 2. 构建监督学习序列（保持原样）
def create_sequences(data, n_steps=5):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps, :4])  # 预测四个特征
    return np.array(X), np.array(y)

# 自适应学习率回调
def adaptive_lr():
    return ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-6,
        verbose=1
    )

def lr_schedule(epoch):
    if epoch < 1000:
        return 0.0001
    else:
        return 0.00005

# 3. LSTM模型构建（保持原样）
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape, return_sequences=False))
    model.add(Dense(4))  # 输出四个特征
    optimizer = Nadam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

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


# 4. 迭代预测函数（修改反归一化方式）
def iterative_predict(model, initial_seq, scaler_lon_lat, predict_steps=30):
    current_seq = initial_seq.copy()
    predicted = []

    for _ in range(predict_steps):
        pred = model.predict(current_seq.reshape(1, *current_seq.shape))[0]
        predicted.append(pred)
        current_seq = np.concatenate([current_seq[1:], pred.reshape(1, -1)])

    # 仅对经纬度进行反归一化
    predicted = np.array(predicted)
    predicted_lonlat = scaler_lon_lat.inverse_transform(predicted[:, :2])
    return predicted_lonlat


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, concatenate
from keras.regularizers import l2
from keras.optimizers import Nadam
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
import tensorflow as tf
def build_encoder_decoder_model(input_shape, units=128):
    encoder_inputs = Input(shape=input_shape)
    
    # 第一层LSTM
    encoder_lstm1 = LSTM(units, return_sequences=True, kernel_regularizer=l2(0.003))(encoder_inputs)
    encoder_lstm1 = Dropout(0.3)(encoder_lstm1)
    
    # 第二层LSTM并获取最终状态
    encoder_output, state_h, state_c = LSTM(units, return_sequences=False, return_state=True, 
                                           kernel_regularizer=l2(0.003))(encoder_lstm1)
    
    # Decoder部分调整
    decoder_input = tf.expand_dims(encoder_output, 1)
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(decoder_input, decoder_input)
    attention = LayerNormalization(epsilon=1e-6)(attention + decoder_input)
    
    decoder_output = Dense(64, activation='relu')(attention)
    decoder_output = Dense(4)(decoder_output)
    decoder_output = tf.squeeze(decoder_output, axis=1)  
    
    model = Model(encoder_inputs, decoder_output)
    optimizer = Nadam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')
    return model
# ---------------------------- 自适应学习率回调（保持原样） ----------------------------
def adaptive_lr():
    return ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-6,
        verbose=1
    )
def lr_schedule(epoch):
    if epoch < 1000:
        return 0.0001
    else:
        return 0.00005
# ---------------------------- 迭代预测函数（需调整以适应新模型） ----------------------------
def iterative_predict(model, initial_seq, scaler_lon_lat, predict_steps=30):
    current_seq = initial_seq.copy()
    predicted = []
    
    for _ in range(predict_steps):
        # 输入形状需符合模型要求（如：样本数=1, 时间步=5, 特征=4）
        pred = model.predict(current_seq[np.newaxis, ...])[0]
        predicted.append(pred)
        # 更新序列：移除第一个时间步，添加新预测（保持特征维度为4）
        current_seq = np.vstack([current_seq[1:], pred])
    
    predicted = np.array(predicted)
    predicted_lonlat = scaler_lon_lat.inverse_transform(predicted[:, :2])
    return predicted_lonlat


# 5. 主程序
if __name__ == "__main__":
    # 参数设置
    n_steps = 5
    batch_size = 32
    epochs = 1000
    predict_steps = 5

    # 数据预处理（获取多个Scaler）
    scaler_lon_lat, scaler_speed, scaler_course, processed_data = preprocess_data('./data/ais_data.csv')

    # 创建序列数据集
    X, y = create_sequences(processed_data, n_steps)

    # 划分训练集和测试集
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 构建并训练模型
    model = build_encoder_decoder_model((X_train.shape[1], X_train.shape[2]))
    callbacks = [
        LearningRateScheduler(lr_schedule),
        # EarlyStopping(monitor='val_loss', patience=25, min_delta=0.0001, restore_best_weights=True)
    ]

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        verbose=1,
                        callbacks=callbacks)

    print("keys:",history.history.keys())
    # 常规预测与评估
    y_pred = model.predict(X_test).squeeze()
    
    # 反归一化处理（仅经纬度）
    y_pred_lonlat = scaler_lon_lat.inverse_transform(y_pred[:, :2]) 
    y_test_lonlat = scaler_lon_lat.inverse_transform(y_test[:, :2])
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_test_lonlat, y_pred_lonlat))
    print(f'Test RMSE: {rmse:.6f} degrees')

    # 迭代预测
    initial_sequence = X_test[0]
    predicted_coords = iterative_predict(model, initial_sequence, scaler_lon_lat, predict_steps)

    # 可视化部分（保持原样）
    actual_steps = min(predict_steps, len(y_test))
    actual_coords = scaler_lon_lat.inverse_transform(y_test[:actual_steps, :2])

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(y_test_lonlat[:, 0], y_test_lonlat[:, 1], 'b-', label='测试集实际')
    plt.plot(y_pred_lonlat[:, 0], y_pred_lonlat[:, 1], 'r--', label='测试集预测')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('测试集表现')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(actual_coords[:, 0], actual_coords[:, 1], 'b-', label='实际轨迹')
    plt.plot(predicted_coords[:, 0], predicted_coords[:, 1], 'r--', label=f'预测{predict_steps}步')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('从测试集起始点迭代预测')
    plt.legend()
    plt.show()
