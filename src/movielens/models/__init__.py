"""
Inicializacion del modulo de modelos.
"""

from .registry import get_model, MODEL_REGISTRY
from .base import BaseModel

__all__ = ['get_model', 'MODEL_REGISTRY', 'BaseModel']
