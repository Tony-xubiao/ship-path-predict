#  基于LSTM的船只轨迹预测系统
![LSTM](https://img.shields.io/badge/Neural_Network-LSTM-blue) ![Keras](https://img.shields.io/badge/Framework-Keras-red) ![Trajectory](https://img.shields.io/badge/Application-Trajectory_Prediction-green)
##  📖 简介
本系统使用LSTM神经网络，基于历史轨迹数据预测船只未来位置和状态：
- **输入**：过去`n_steps`个时间步的`longitude`, `latitude`, `speed`, `course`数据
- **输出**：预测下一个时间步的上述四个参数
- **特点**：支持迭代预测无限时长（注意误差会随时间累积）
##  🚀 快速开始
### 前置条件
- Python 3.9+
- 依赖库：`pip install -r requirements.txt`

### 使用流程
1. **数据准备**
   - 准备CSV格式数据（参考`data/ais_data.csv`）
   - 必须包含字段：`longitude`, `latitude`, `speed`, `course` 
2. **训练模型**
   ```bash
   python train.py 
3. **推理模型**
   ```bash
   python predict.py

3. **(可选）推理模型**
   ```bash
   python finetune.py
### 配置参数


|  参数   | 默认值  | 描述  |
|  ----  | ----  | ----  |
| n_steps  | 20 |时间窗口大小（历史步数） |
| batch_size  | 32 |训练批处理量 |
| epochs  | 2000 |训练轮数 |
| predict_steps  | 5 |迭代预测的时间步数  |  

### 💡 附加功能
**数据可视化**    
&emsp;运行`data/data_show.py`查看轨迹图  
**模拟数据**    
&emsp;使用`data/data_gen.py`生成测试数据  
**全流程演示**  
&emsp;`demo.py`包含训练、预测、可视化完整流程

### 📂 项目结构
```
demo_default  
--data                          # 数据相关  
    --ais_data.csv                  # ais数据,网上找的真实数据，用于训练  
    --data_gen.py                   # 数据生成，脚本生成假的AIS数据   
    --data_show.py                  # 数据可视化，展示轨迹图  
    --validate_data.csv             # 单独准备的数据，用于验证模型准确性  
--checkpoint                    # 模型保存目录  
    --2025-07-23-15-05-12           # 模型保存目录-按照时间命名  
        --model.h5                      # 模型保存文件  
        --scaler_course.pkl             # 归一化参数保存文件，角度归一化      
        --scaler_lon_lat.pkl            # 归一化参数保存文件，经纬度归一化  
        --scaler_speed.pkl              # 归一化参数保存文件，速度归一化  
-demo.py                            # 模型训练、推理、可视化，全流程集中在这个文件，学习、演示用    
-train.py                           # 模型训练、保存模型checkpoint  
-finetune.py                        # 加载模型、模型微调、保存模型checkpoint  
-predict.py                         # 模型推理  
-readme.md                          # 说明  
```
### 📜 许可证
本项目采用
[MIT License](https://opensource.org/license/mit )
