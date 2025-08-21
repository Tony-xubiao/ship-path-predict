# -*- coding: utf-8 -*-
import paramiko
from fastapi import APIRouter

import shipai.model.model_load as ml
import numpy as np

from shipai.model.api_param import PredictReq
from shipai.predict import preprocess_validation_data, iterative_predict, load_model_scaler

router = APIRouter()

@router.post("/exec")
async def predict_exec(param: PredictReq):
    model_code = str(param.model_code)
    print(f'going to predict...modelCode:{model_code}')
    mmsi = param.mmsi
    steps = param.steps

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ml.SFTP_HOST, username=ml.SFTP_USERNAME, password=ml.SFTP_PASSWORD)
    sftp = ssh.open_sftp()
    try:
        # 加载模型、scaler、验证数据
        model, scaler_lon_lat, scaler_speed, scaler_course = load_model_scaler(mmsi, model_code)
        print(f'model|scaler|validate data LOADED...')
        validate_data_path = ml.validate_data_path(mmsi, model_code)
        predict_result = predict_future_trajectory(
            sftp=sftp,
            model=model,
            scaler_lon_lat=scaler_lon_lat,
            scaler_speed=scaler_speed,
            scaler_course=scaler_course,
            validate_data_path=validate_data_path,
            input_start=40,
            n_steps=20,
            predict_steps=steps
        )
        if np.isnan(predict_result).any():
            # data_arr = [[1,11], [2,22], [3,33], [4,44], [5,55], [6,66], [7,77], [8,88], [9,99], [10,110], [11,111]]
            # return {"message": "success", "result": 0, "data": data_arr}
            raise ValueError("预测结果包含NaN值！可能的原因："
                             "\n1. 模型训练不充分"
                             "\n2. 输入数据异常"
                             "\n3. 标准化器与数据不匹配"
                             "\n4. 梯度爆炸导致数值溢出")
        print(f'predict logic EXECUTED...')
        if predict_result is not None:
            predict_result_serializable = predict_result.tolist()
    except Exception as e:
        print(e)
        return {"message": str(e), "result": -1}
    finally:
        sftp.close()
    return {"message": "success", "result": 0, "data": predict_result_serializable}

def predict_future_trajectory(
        sftp,
        model,
        scaler_lon_lat,
        scaler_speed,
        scaler_course,
        validate_data_path,
        input_start,
        n_steps,
        predict_steps
):
    """
    预测未来指定时间步内的航行轨迹

    参数:
        model: 加载的预测模型
        scaler_lon_lat: 经度纬度归一化器
        scaler_speed: 速度归一化器
        scaler_course: 航向归一化器
        validate_data_path: 验证数据文件路径
        input_start: 从验证数据的何处开始取输入序列
        n_steps: 输入序列的时间步长(需与训练时一致)
        predict_steps: 要预测的未来时间步数

    返回:
        dict: 包含以下键值:
            - 'input_coords': 输入序列的真实坐标(形状[n_steps, 2])
            - 'actual_coords': 整个验证集的真实坐标
            - 'predicted_coords': 预测的未来轨迹坐标(形状[predict_steps, 2])
    """
    print(f'go in predict logic...')
    # 处理数据并进行预测
    scaled_data, raw_features = preprocess_validation_data(
        sftp,
        validate_data_path,
        scaler_lon_lat,
        scaler_speed,
        scaler_course
    )

    # 检查数据长度是否足够
    if len(scaled_data) < input_start + n_steps + predict_steps:
        raise ValueError(
            f"data length is not enough.  You need at least {input_start + n_steps + predict_steps} data points，"
            f"but get only {len(scaled_data)} available."
        )

    # 创建输入序列
    initial_sequence = scaled_data[input_start: input_start + n_steps]

    # 进行迭代预测
    return iterative_predict(model, initial_sequence, scaler_lon_lat, predict_steps)