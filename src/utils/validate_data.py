from typing import List, Tuple
import pandas as pd


def basic_schema_checks(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, list]:
    missing = [c for c in required_columns if c not in df.columns]
    return (len(missing) == 0, missing)


def sanity_checks(df: pd.DataFrame) -> Tuple[bool, list]:
    issues = []
    if df.empty:
        issues.append("DataFrame vide")
    if "TotalCharges" in df.columns and df["TotalCharges"].isna().mean() > 0.2:
        issues.append("Trop de NaN dans TotalCharges")
    if "tenure" in df.columns and (df["tenure"] < 0).any():
        issues.append("tenure contient des valeurs négatives")
    return (len(issues) == 0, issues)


