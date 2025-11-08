"""Clase base para clasificadores."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseClassifier(ABC):
    """Clase base abstracta para clasificadores multi-label."""
    
    def __init__(self, **kwargs):
        self.hyperparameters = kwargs
        self._model = None
    
    @abstractmethod
    def build(self) -> Any:
        """Construir y retornar instancia del clasificador."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Retornar nombre del algoritmo."""
        pass
    
    @property
    def params(self) -> Dict[str, Any]:
        """Retornar hiperparametros para logging en MLflow."""
        return self.hyperparameters
    
    def get_model(self) -> Any:
        """Obtener instancia del modelo construido."""
        if self._model is None:
            self._model = self.build()
        return self._model
