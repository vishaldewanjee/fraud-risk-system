from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "artifacts" / "fraud_model.joblib"
METADATA_PATH = BASE_DIR / "artifacts" / "model_metadata.json"


# --------------------------------------------------
# Load model and metadata
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)


FEATURE_NAMES = metadata["feature_names"]
THRESHOLD = metadata["threshold"]


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Fraud Risk Detection API",
    description="API for predicting credit card transaction fraud risk.",
    version="1.0.0"
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class Transaction(BaseModel):
    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Fraud Risk API is running!"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------


@app.post("/predict")
def predict(transaction: Transaction):

    data = transaction.model_dump()

    # Feature engineering must match model training
    data["log_amount"] = np.log1p(data["Amount"])

    # Build model input in the expected order
    input_data = pd.DataFrame(
        [[data[feature] for feature in FEATURE_NAMES]],
        columns=FEATURE_NAMES
    )

    # Debugging information
    print("V1 received:", data["V1"])
    print("Amount received:", data["Amount"])
    print("Time received:", data["Time"])
    print("Input shape:", input_data.shape)
    print("Input columns:", input_data.columns.tolist())

    # Get probability
    fraud_probability = float(
        model.predict_proba(input_data)[0][1]
    )

    print("V1 received:", data["V1"])
    print("Amount received:", data["Amount"])
    print("Raw fraud probability:", fraud_probability)


    prediction = int(fraud_probability >= THRESHOLD)

    if fraud_probability < 0.20:
        risk_level = "Low"
        decision = "Approve"
    elif fraud_probability < 0.60:
        risk_level = "Medium"
        decision = "Monitor"
    elif fraud_probability < 0.80:
        risk_level = "High"
        decision = "Manual Review"
    else:
        risk_level = "Critical"
        decision = "Block Transaction"

    return {
        "fraud_probability": fraud_probability,
        "fraud_prediction": prediction,
        "risk_level": risk_level,
        "decision": decision
    }
