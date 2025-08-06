from fastapi import FastAPI
from shipai.api.api_predict import router as strategy_router
from shipai.api.api_train import router as train_router

app = FastAPI()


app.include_router(strategy_router, prefix="/predict", tags=["推理相关接口"])
app.include_router(train_router, prefix="/train", tags=["训练相关接口"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)