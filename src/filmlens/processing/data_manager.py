"""Data loading, saving, and pipeline management."""
import pandas as pd
from pathlib import Path
import joblib
from typing import List
import logging
from sklearn.pipeline import Pipeline

from filmlens.config import DATASET_DIR, TRAINED_MODEL_DIR, config
from filmlens import __version__ as _version

logger = logging.getLogger(__name__)


def load_dataset(file_name: str = None) -> pd.DataFrame:
    """
    Load dataset from CSV.
    
    Args:
        file_name: CSV filename relative to DATASET_DIR
        
    Returns:
        DataFrame with loaded data
    """
    if file_name is None:
        file_name = config.data_config.train_data_file
    
    file_path = DATASET_DIR / file_name
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df):,} rows from {file_path.name}")
    
    return df


def pre_pipeline_preparation(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data before pipeline processing.
    
    Args:
        data_frame: Raw dataframe
        
    Returns:
        Cleaned dataframe
    """
    df = data_frame.copy()
    
    # Drop rows with missing values in critical columns
    drop_cols = config.data_config.drop_na_columns
    if drop_cols:
        initial_len = len(df)
        df = df.dropna(subset=drop_cols)
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped:,} rows with missing values")
    
    # Add basic features
    text_col = config.data_config.text_column
    if 'description_length' not in df.columns:
        df['description_length'] = df[text_col].str.len()
    
    if 'word_count' not in df.columns:
        df['word_count'] = df[text_col].str.split().str.len()
    
    logger.info(f"Pre-pipeline prep completed: {len(df):,} rows")
    
    return df


def save_pipeline(pipeline_to_persist: Pipeline, algorithm_name: str = None) -> Path:
    """
    Guardar pipeline entrenado en disco.
    
    Args:
        pipeline_to_persist: Pipeline de sklearn entrenado
        algorithm_name: Nombre del algoritmo (opcional)
        
    Returns:
        Path del archivo guardado
    """
    algo_name = algorithm_name or config.model.algorithm
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}_{algo_name}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name
    
    TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(pipeline_to_persist, save_path)
    logger.info(f"Pipeline guardado en {save_path}")
    
    return save_path


def load_pipeline(file_name: str = None) -> Pipeline:
    """
    Load persisted pipeline from disk.
    
    Args:
        file_name: Optional specific filename to load
        
    Returns:
        Loaded sklearn pipeline
    """
    if file_name is None:
        file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    
    file_path = TRAINED_MODEL_DIR / file_name
    
    if not file_path.exists():
        raise FileNotFoundError(f"Pipeline not found: {file_path}")
    
    trained_model = joblib.load(filename=file_path)
    logger.info(f"Pipeline loaded from {file_path.name}")
    
    return trained_model


def remove_old_pipelines(files_to_keep: List[str] = None) -> None:
    """
    Remove old pipelines, keeping only specified versions.
    
    Args:
        files_to_keep: List of filenames to preserve
    """
    if files_to_keep is None:
        files_to_keep = [f"{config.app_config.pipeline_save_file}{_version}.pkl"]
    
    do_not_delete = files_to_keep + ["__init__.py"]
    
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
            logger.info(f"Removed old pipeline: {model_file.name}")
