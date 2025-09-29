import pandas as pd


def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Encodage binaire déterministe pour les colonnes à 2 modalités.
    """
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")
    if valset == {"Male", "Female"}:
        return s.map({"Female": 0, "Male": 1}).astype("Int64")

    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")

    return s


def build_features(df: pd.DataFrame, target_col: str = "Churn", serving: bool = False) -> pd.DataFrame:
    """
    Pipeline de feature engineering pour l'entraînement et le serving.

    - en mode serving=True, on tente l'encodage binaire pour TOUTES les colonnes object
      connues (Yes/No, Male/Female), même si nunique == 1, afin d'éviter de
      laisser des dtypes object passer au modèle.
    """
    df = df.copy()

    # Colonnes object hors cible
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]

    # Encodage binaire
    if serving:
        # Essayer d'encoder toutes les colonnes object connues
        for c in obj_cols:
            mapped = _map_binary_series(df[c].astype(str))
            # Si mapping a produit une série entière nullable, on remplace
            if mapped.dtype.name == "Int64" or pd.api.types.is_integer_dtype(mapped):
                df[c] = mapped
    else:
        # Entraînement: basé sur la cardinalité
        binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
        for c in binary_cols:
            df[c] = _map_binary_series(df[c].astype(str))

    # Colonnes bool -> int
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    # Après encodage binaire, one-hot pour le reste des colonnes object
    remaining_obj = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    if remaining_obj:
        df = pd.get_dummies(df, columns=remaining_obj, drop_first=True)

    # Convertir les Int64 (nullable) en int
    for c in df.columns:
        if str(df[c].dtype) == "Int64":
            df[c] = df[c].fillna(0).astype(int)

    return df


