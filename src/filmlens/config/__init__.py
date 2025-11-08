"""Configuration management using Pydantic."""

from pathlib import Path
from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field
import yaml
import logging

logger = logging.getLogger(__name__)

# Paths
PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # src/filmlens
ROOT = PACKAGE_ROOT.parent.parent  # project root
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = ROOT / "data"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

# Ensure directories exist
TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)


class AppConfig(BaseModel):
    """Application configuration."""
    package_name: str
    pipeline_save_file: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str


class DataConfig(BaseModel):
    """Data loading configuration."""
    train_data_file: str
    text_column: str
    drop_na_columns: List[str]
    test_size: float = Field(gt=0, lt=1)
    random_state: int = 42
    stratify_by: str = None
    target_columns: List[str]


class TFIDFConfig(BaseModel):
    """TF-IDF vectorization parameters."""
    max_features: int = Field(gt=0)
    ngram_range: List[int]
    min_df: Union[int, float]
    max_df: Union[int, float]
    stop_words: str


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""
    tfidf: TFIDFConfig
    keyword_categories: List[str]
    use_genre_features: bool


class ModelConfig(BaseModel):
    """Model configuration."""
    algorithm: str = Field(
        default="random_forest",
        description="Nombre del algoritmo seleccionado"
    )
    algorithms: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Hiperparametros por algoritmo"
    )
    
    def get_algorithm_params(self, algorithm_name: str = None) -> Dict[str, Any]:
        """Obtener hiperparametros del algoritmo seleccionado."""
        algo = algorithm_name or self.algorithm
        return self.algorithms.get(algo, {})


class MetricsConfig(BaseModel):
    """Evaluation metrics configuration."""
    primary_metric: str
    additional_metrics: List[str]


class Config(BaseModel):
    """Master configuration object."""
    app_config: AppConfig
    data_config: DataConfig
    feature_config: FeatureConfig
    model: ModelConfig
    metrics_config: MetricsConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise FileNotFoundError(f"Config not found at {CONFIG_FILE_PATH}")


def fetch_config_from_yaml(cfg_path: Path = None) -> Dict:
    """Parse YAML config file."""
    if not cfg_path:
        cfg_path = find_config_file()
    
    with open(cfg_path, 'r') as f:
        parsed_config = yaml.safe_load(f)
    return parsed_config


def create_and_validate_config(parsed_config: Dict = None) -> Config:
    """Validate config with Pydantic."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()
    
    _config = Config(**parsed_config)
    return _config


# Load config on module import
config = create_and_validate_config()

logger.info(f"Config loaded: {config.app_config.package_name}")
