import paramiko
from fastapi import APIRouter

import shipai.model.model_load as ml
from shipai.model.api_param import TrainReq
from shipai.train import train

router = APIRouter()

@router.post("/exec")
def train_exec(param: TrainReq):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ml.SFTP_HOST, username=ml.SFTP_USERNAME, password=ml.SFTP_PASSWORD)
    sftp = ssh.open_sftp()
    try:
        data_volume = train(sftp, param.mmsi, param.model_code)
    except Exception as e:
        print(e)
        return {"message": str(e), "result": -1}
    finally:
        sftp.close()
    return {"message": "success", "result": 0, "data_volume": data_volume}