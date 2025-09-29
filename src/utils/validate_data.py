from typing import List, Tuple, Dict, Any
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


def validate_payload_records(records: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Valide la présence et les types/valeurs des variables brutes indispensables
    pour construire les features attendues par le modèle.

    Retourne (ok, erreurs[])
    """
    required_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]

    errors: List[str] = []

    if not records:
        return False, ["payload vide: 'instances' est requis et non vide"]

    # Vérif colonnes requises
    missing_any = set(required_columns) - set(records[0].keys())
    if missing_any:
        errors.append(f"colonnes manquantes: {sorted(list(missing_any))}")

    # Vérifs par enregistrement
    yes_no_fields = {
        "Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "PaperlessBilling"
    }
    for idx, rec in enumerate(records):
        # Types numériques de base
        for num_field in ["tenure", "MonthlyCharges", "TotalCharges"]:
            if num_field in rec:
                try:
                    float(rec[num_field])
                except Exception:
                    errors.append(f"instance {idx}: champ {num_field} doit être numérique")

        # SeniorCitizen doit être 0/1
        if "SeniorCitizen" in rec and rec["SeniorCitizen"] not in [0, 1, "0", "1"]:
            errors.append(f"instance {idx}: SeniorCitizen doit être 0/1")

        # Yes/No
        for f in yes_no_fields:
            if f in rec and str(rec[f]) not in ["Yes", "No", "No phone service"]:
                errors.append(f"instance {idx}: {f} doit être 'Yes'/'No' (ou 'No phone service' si applicable)")

        # Domaines restants (non bloquants, mais on signale)
        if "gender" in rec and str(rec["gender"]) not in ["Male", "Female"]:
            errors.append(f"instance {idx}: gender doit être 'Male'/'Female'")

    return (len(errors) == 0, errors)


