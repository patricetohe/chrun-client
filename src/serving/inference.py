import os
import joblib
import pandas as pd


def load_feature_columns(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_model(model_dir: str):
    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")
    return joblib.load(model_path)


def prepare_inference_frame(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    X = df.reindex(columns=feature_columns, fill_value=0)
    return X


def predict_proba(model, X: pd.DataFrame) -> pd.Series:
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=X.index)


