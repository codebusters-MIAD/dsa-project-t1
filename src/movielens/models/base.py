"""
Clase base para modelos de clasificacion.
"""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Clase base abstracta para modelos de clasificacion.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
    
    @abstractmethod
    def build(self):
        """Construye el modelo con la configuracion especificada"""
        pass
    
    def fit(self, X_train, Y_train):
        """Entrena el modelo"""
        if self.model is None:
            self.build()
        self.model.fit(X_train, Y_train)
    
    def predict(self, X):
        """Realiza predicciones"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Realiza predicciones probabilisticas"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        raise NotImplementedError("Modelo no soporta predict_proba")
