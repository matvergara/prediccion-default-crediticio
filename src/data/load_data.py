import pandas as pd
from pathlib import Path

# Raíz del proyecto: dos niveles arriba de src/data/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / 'data' / 'raw' / 'credit_defaults.csv'
PROCESSED_PATH = PROJECT_ROOT / 'data' / 'processed' / 'credit_defaults_clean.csv'


def load_raw_data() -> pd.DataFrame:
    """
    Carga datos crudos desde data/raw.

    Returns:
        DataFrame con datos crudos.
    """
    return pd.read_csv(RAW_PATH, sep=";", header=1)


def save_processed_data(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    """
    Guarda el DataFrame preprocesado en data/processed.

    Args:
        df: DataFrame preprocesado.
        path: Ruta de destino (por defecto data/processed/credit_defaults_clean.csv).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_processed_data() -> pd.DataFrame:
    """
    Carga el dataset preprocesado desde data/processed.

    Returns:
        DataFrame preprocesado.
    """
    return pd.read_csv(PROCESSED_PATH)
