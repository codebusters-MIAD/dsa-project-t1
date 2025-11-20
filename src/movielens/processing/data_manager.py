"""
Gestion de carga y preparacion de datos para el modelo de sensibilidad.
"""

import pandas as pd
import csv
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def leer_csv_robusto(path: str) -> pd.DataFrame:
    """
    Carga un archivo CSV con manejo robusto de delimitadores y encoding.
    Intenta multiples estrategias para garantizar una carga exitosa.
    """
    candidatos = [';', ',', '\t', '|']
    
    for sep in candidatos:
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                engine='python',
                encoding='utf-8-sig',
                quotechar='"',
                doublequote=True,
            )
            return df
        except Exception:
            pass
    
    try:
        with open(path, "r", encoding="utf-8-sig", newline='') as f:
            muestra = f.read(4096)
            dialecto = csv.Sniffer().sniff(muestra, delimiters=[',', ';', '\t', '|'])
            sep_detectado = dialecto.delimiter
        
        df = pd.read_csv(
            path,
            sep=sep_detectado,
            engine='python',
            encoding='utf-8-sig',
            quotechar='"',
            doublequote=True,
        )
        return df
    except Exception:
        pass
    
    try:
        df = pd.read_csv(
            path,
            sep=None,
            engine='python',
            encoding='utf-8-sig',
            quotechar='"',
            doublequote=True,
            on_bad_lines='skip'
        )
        return df
    except Exception as e:
        raise RuntimeError(f"No fue posible cargar el CSV: {e}")


def load_dataset(config: dict) -> pd.DataFrame:
    """
    Carga el dataset de sensibilidad desde la ruta configurada.
    """
    base_path = Path(__file__).resolve().parents[3]
    dataset_path = base_path / config['data_config']['dataset_path']
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")
    
    df = leer_csv_robusto(str(dataset_path))
    return df


def balance_dataset(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Balancea el dataset usando oversampling en clases minoritarias.
    Se balancea sobre la columna target especificada en la configuracion.
    """
    if not config['balance_config']['enable_balancing']:
        return df
    
    target_col = config['balance_config']['target_column']
    balance_ratio = config['balance_config']['balance_ratio']
    
    df_major = df[df[target_col] == 'sin_contenido']
    df_med = df[df[target_col] == 'moderado']
    df_high = df[df[target_col] == 'alto']
    
    n_samples_target = int(len(df_major) * balance_ratio)
    
    df_med_upsampled = resample(
        df_med, 
        replace=True, 
        n_samples=n_samples_target, 
        random_state=config['data_config']['random_state']
    )
    
    df_high_upsampled = resample(
        df_high, 
        replace=True, 
        n_samples=n_samples_target, 
        random_state=config['data_config']['random_state']
    )
    
    df_balanced = pd.concat([df_major, df_med_upsampled, df_high_upsampled])
    df_balanced = df_balanced.sample(
        frac=1, 
        random_state=config['data_config']['random_state']
    ).reset_index(drop=True)
    
    return df_balanced


def split_data(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset en conjuntos de entrenamiento y prueba.
    """
    test_size = config['data_config']['test_size']
    random_state = config['data_config']['random_state']
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline completo de preparacion de datos: carga, balanceo y division.
    """
    df = load_dataset(config)
    df = balance_dataset(df, config)
    train_df, test_df = split_data(df, config)
    
    return train_df, test_df
