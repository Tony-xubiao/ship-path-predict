import paramiko
from fastapi import APIRouter

import shipai.model.model_load as ml

from shipai.model.api_param import PredictReq
from shipai.predict import preprocess_validation_data, iterative_predict, load_model_scaler

router = APIRouter()

@router.post("/exec")
async def predict_exec(param: PredictReq):
    model_code = param.model_code
    mmsi = param.mmsi
    steps = param.steps

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ml.SFTP_HOST, username=ml.SFTP_USERNAME, password=ml.SFTP_PASSWORD)
    sftp = ssh.open_sftp()
    try:
        # ����ģ�͡�scaler����֤����
        model, scaler_lon_lat, scaler_speed, scaler_course = load_model_scaler(mmsi, model_code)
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
    except Exception as e:
        print(e)
        raise
    finally:
        sftp.close()
    return predict_result

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
    Ԥ��δ��ָ��ʱ�䲽�ڵĺ��й켣

    ����:
        model: ���ص�Ԥ��ģ��
        scaler_lon_lat: ����γ�ȹ�һ����
        scaler_speed: �ٶȹ�һ����
        scaler_course: �����һ����
        validate_data_path: ��֤�����ļ�·��
        input_start: ����֤���ݵĺδ���ʼȡ��������
        n_steps: �������е�ʱ�䲽��(����ѵ��ʱһ��)
        predict_steps: ҪԤ���δ��ʱ�䲽��

    ����:
        dict: �������¼�ֵ:
            - 'input_coords': �������е���ʵ����(��״[n_steps, 2])
            - 'actual_coords': ������֤������ʵ����
            - 'predicted_coords': Ԥ���δ���켣����(��״[predict_steps, 2])
    """

    # �������ݲ�����Ԥ��
    scaled_data, raw_features = preprocess_validation_data(
        sftp,
        validate_data_path,
        scaler_lon_lat,
        scaler_speed,
        scaler_course
    )

    # ������ݳ����Ƿ��㹻
    if len(scaled_data) < input_start + n_steps + predict_steps:
        raise ValueError(
            f"���ݳ��Ȳ��㡣��Ҫ���� {input_start + n_steps + predict_steps} �����ݵ㣬"
            f"��ֻ�� {len(scaled_data)} �����á�"
        )

    # ������������
    initial_sequence = scaled_data[input_start: input_start + n_steps]

    # ���е���Ԥ��
    return iterative_predict(model, initial_sequence, scaler_lon_lat, predict_steps)