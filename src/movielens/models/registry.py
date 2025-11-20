"""
Registro de modelos disponibles.
"""

from .logistic_regression import LogisticRegressionModel
from .random_forest import RandomForestModel
from .lightgbm import LightGBMModel


MODEL_REGISTRY = {
    'logistic_regression': LogisticRegressionModel,
    'random_forest': RandomForestModel,
    'lightgbm': LightGBMModel
}


def get_model(algorithm: str, config: dict):
    """
    Obtiene una instancia del modelo segun el algoritmo especificado.
    """
    if algorithm not in MODEL_REGISTRY:
        raise ValueError(
            f"Algoritmo '{algorithm}' no soportado. "
            f"Opciones: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_class = MODEL_REGISTRY[algorithm]
    return model_class(config)
