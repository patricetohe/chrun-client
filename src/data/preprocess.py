import pandas as pd


def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Nettoyage de base pour le dataset Telco churn.
    - nettoyage des entêtes
    - suppression IDs évidents
    - conversion TotalCharges en numérique
    - mapping Churn vers 0/1 si nécessaire
    - gestion simple des NA
    """
    df = df.copy()

    # tidy headers
    df.columns = df.columns.str.strip()

    # drop ids si présents
    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # target vers 0/1 si Yes/No
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    # TotalCharges -> float
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # SeniorCitizen -> int 0/1
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    # NA:
    # - num: 0
    # - autres: laisser pour one-hot
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df


