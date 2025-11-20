"""
Modelo de Random Forest para clasificacion multi-output.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from .base import BaseModel


class RandomForestModel(BaseModel):
    """
    Modelo de Random Forest con MultiOutputClassifier.
    """
    
    def build(self):
        """Construye el modelo con configuracion de config.yml"""
        cfg = self.config['model_config']['random_forest']
        
        base_clf = RandomForestClassifier(
            n_estimators=cfg['n_estimators'],
            max_depth=cfg['max_depth'],
            random_state=cfg['random_state'],
            n_jobs=cfg['n_jobs'],
            class_weight=cfg['class_weight']
        )
        
        self.model = MultiOutputClassifier(base_clf)
        return self.model
