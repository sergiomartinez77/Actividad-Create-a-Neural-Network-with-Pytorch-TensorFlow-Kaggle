"""
Preprocesamiento para datos tabulares del dataset Breast Cancer Wisconsin.
"""
import numpy as np
import pandas as pd
from typing import Tuple

def preprocess_tabular(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Normaliza un DataFrame para datos tabulares.
    Si incluye 'target', la separa.
    Retorna (X, y) o (X, None) si no hay target.
    """
    df = df.copy()

    y = None
    if "target" in df.columns:
        y = df.pop("target").values

    # Normalización 0-1
    X = (df - df.min()) / (df.max() - df.min() + 1e-8)
    return X.values.astype(np.float32), y
