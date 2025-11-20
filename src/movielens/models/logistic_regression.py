"""
Modelo de Regresion Logistica para clasificacion multi-output.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from .base import BaseModel


class LogisticRegressionModel(BaseModel):
    """
    Modelo de Regresion Logistica con MultiOutputClassifier.
    """
    
    def build(self):
        """Construye el modelo con configuracion de config.yml"""
        cfg = self.config['model_config']['logistic_regression']
        
        base_clf = LogisticRegression(
            solver=cfg['solver'],
            max_iter=cfg['max_iter'],
            random_state=cfg['random_state']
        )
        
        self.model = MultiOutputClassifier(base_clf)
        return self.model
