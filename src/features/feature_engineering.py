"""
Módulo de feature engineering para el dataset de crédito.
"""

import pandas as pd
from typing import List


def preprocesar_meses_deuda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Descompone la columna meses_deuda_sep en features interpretables.

    La codificación original mezcla estados cualitativos (-2, -1, 0) con
    una escala numérica de meses de atraso (1+), lo que no es directamente
    utilizable por los modelos. Esta función los separa en variables binarias
    y una variable numérica limpia.

    Args:
        df: DataFrame con la columna 'meses_deuda_sep'.

    Returns:
        Copia del DataFrame con las nuevas features y sin la columna original.
    """
    df = df.copy()

    df['tiene_deuda_sep'] = (df['meses_deuda_sep'] >= 1).astype(int)
    df['n_meses_deuda_sep'] = df['meses_deuda_sep'].clip(lower=0)
    df['sin_uso_sep'] = (df['meses_deuda_sep'] == -2).astype(int)
    df['pago_minimo_sep'] = (df['meses_deuda_sep'] == 0).astype(int)
    df['pago_completo_sep'] = (df['meses_deuda_sep'] == -1).astype(int)

    df.drop(columns=['meses_deuda_sep'], inplace=True)
    return df


def crear_features_avanzadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea ratios financieros que relacionan variables existentes.

    Args:
        df: DataFrame con columnas 'factura_sep', 'limite_credito' y 'pago_sep'.

    Returns:
        Copia del DataFrame con las nuevas features de ratio.
    """
    df = df.copy()

    df['ratio_uso_sep'] = (df['factura_sep'] / df['limite_credito']).clip(0, 1)
    df['ratio_pago_sep'] = (df['pago_sep'] / (df['factura_sep'] + 1)).clip(0, 2)

    return df


def aplicar_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering.

    Args:
        df: DataFrame preprocesado (salida de preprocess.preprocesar_datos).

    Returns:
        DataFrame con todas las features listas para modelado.
    """
    df = preprocesar_meses_deuda(df)
    df = crear_features_avanzadas(df)
    return df


def get_feature_cols() -> dict:
    """
    Retorna las listas de features numéricas y categóricas para el modelado.

    Returns:
        Diccionario con claves 'numericas' y 'categoricas'.
    """
    return {
        'numericas': [
            'limite_credito', 'edad', 'pago_sep', 'factura_sep',
            'n_meses_deuda_sep', 'ratio_uso_sep', 'ratio_pago_sep'
        ],
        'categoricas': ['genero', 'educacion', 'estado_civil']
    }
