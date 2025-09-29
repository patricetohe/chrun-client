from fastapi import FastAPI
from pydantic import BaseModel, Field, conint, confloat
from typing import List, Any, Literal
import os
import sys
import pandas as pd

# rendre src importable si lancé depuis project root
# _ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if _ROOT_DIR not in sys.path:
#     sys.path.insert(0, _ROOT_DIR)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.serving.inference import load_model, load_feature_columns, prepare_inference_frame, predict_proba
from src.utils.validate_data import validate_payload_records


app = FastAPI(title="Churn ML API", version="1.0.0")


class Record(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: conint(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: conint(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: confloat(ge=0)
    TotalCharges: confloat(ge=0)


class PredictRequest(BaseModel):
    instances: List[Record]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(body: PredictRequest, model_dir: str = os.path.join("artifacts", "model")) -> Any:
    data = [r.dict() for r in body.instances]
    df = pd.DataFrame(data)

    # Prétraitement + features (serving=True pour éviter dtypes object)
    df = preprocess_data(df, target_col="Churn")
    df_feat = build_features(df, target_col="Churn", serving=True)

    # Chargement modèle & colonnes
    model = load_model(model_dir)
    feature_cols = load_feature_columns(os.path.join(model_dir, "feature_columns.txt"))
    X = prepare_inference_frame(df_feat, feature_cols)

    proba = predict_proba(model, X)
    return {"predictions": proba.round(6).tolist()}

# Depuis churn-ml
# python -m uvicorn src.app.main:app --reload

# Docs interactives:
# http://127.0.0.1:8000/docs
