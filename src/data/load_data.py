import os
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Charger un fichier CSV dans un DataFrame pandas.

    Args:
        file_path: chemin vers le fichier CSV.

    Returns:
        DataFrame pandas contenant les données chargées.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable: {file_path}")

    return pd.read_csv(file_path)


