"""
Modelo LightGBM para clasificacion multi-output.
"""

from sklearn.multioutput import MultiOutputClassifier
from .base import BaseModel

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class LightGBMModel(BaseModel):
    """
    Modelo LightGBM con MultiOutputClassifier.
    Fallback a Logistic Regression si LightGBM no esta disponible.
    """
    
    def build(self):
        """Construye el modelo con configuracion de config.yml"""
        if not LIGHTGBM_AVAILABLE:
            print("LightGBM no disponible, usando Logistic Regression")
            from sklearn.linear_model import LogisticRegression
            base_clf = LogisticRegression(solver='liblinear', max_iter=1000)
        else:
            cfg = self.config['model_config']['lightgbm']
            base_clf = LGBMClassifier(
                n_estimators=cfg['n_estimators'],
                max_depth=cfg['max_depth'],
                learning_rate=cfg['learning_rate'],
                random_state=cfg['random_state'],
                n_jobs=cfg['n_jobs'],
                class_weight=cfg['class_weight'],
                verbose=cfg['verbose']
            )
        
        self.model = MultiOutputClassifier(base_clf)
        return self.model
