import io

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model import run_prediction


app = FastAPI(title="MLCF Stock Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lookback: int = Form(20),
    test_ratio: float = Form(0.2),
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        return run_prediction(df, lookback=lookback, test_ratio=test_ratio)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
