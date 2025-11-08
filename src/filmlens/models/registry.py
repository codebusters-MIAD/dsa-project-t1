from typing import Dict, Type, List
from .base import BaseClassifier


class ModelRegistry:
    """Registro para almacenar y recuperar clases de modelos."""
    
    _models: Dict[str, Type[BaseClassifier]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorador para registrar una clase de modelo."""
        def decorator(model_class: Type[BaseClassifier]):
            if name in cls._models:
                raise ValueError(f"Modelo '{name}' ya registrado")
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get(cls, name: str, **kwargs) -> BaseClassifier:
        """Obtener instancia de modelo por nombre."""
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(
                f"Modelo '{name}' no encontrado. "
                f"Disponibles: {available}"
            )
        return cls._models[name](**kwargs)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """Listar todos los modelos registrados."""
        return list(cls._models.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Verificar si un modelo esta registrado."""
        return name in cls._models
