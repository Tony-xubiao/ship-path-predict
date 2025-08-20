from pydantic import BaseModel

class PredictReq(BaseModel):
    mmsi: str
    model_code: str
    steps: int

class TrainReq(BaseModel):
    mmsi: str
    model_code: str