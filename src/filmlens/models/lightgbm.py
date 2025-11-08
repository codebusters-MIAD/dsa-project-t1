from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier

from .base import BaseClassifier
from .registry import ModelRegistry


@ModelRegistry.register("lightgbm")
class LightGBMModel(BaseClassifier):
    """Clasificador LightGBM multi-label."""
    
    @property
    def name(self) -> str:
        return "LightGBM"
    
    def build(self) -> MultiOutputClassifier:
        lgbm = LGBMClassifier(
            n_estimators=self.hyperparameters.get('n_estimators', 100),
            learning_rate=self.hyperparameters.get('learning_rate', 0.1),
            max_depth=self.hyperparameters.get('max_depth', -1),
            num_leaves=self.hyperparameters.get('num_leaves', 31),
            random_state=self.hyperparameters.get('random_state', 42),
            n_jobs=self.hyperparameters.get('n_jobs', -1),
            verbose=self.hyperparameters.get('verbose', -1),
            force_col_wise=True
        )
        return MultiOutputClassifier(lgbm, n_jobs=-1)
