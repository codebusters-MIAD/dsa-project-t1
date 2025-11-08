"""Implementacion de Random Forest."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from .base import BaseClassifier
from .registry import ModelRegistry


@ModelRegistry.register("random_forest")
class RandomForestModel(BaseClassifier):
    """Clasificador Random Forest multi-label."""
    
    @property
    def name(self) -> str:
        return "RandomForest"
    
    def build(self) -> MultiOutputClassifier:
        rf = RandomForestClassifier(
            n_estimators=self.hyperparameters.get('n_estimators', 100),
            max_depth=self.hyperparameters.get('max_depth', None),
            min_samples_split=self.hyperparameters.get('min_samples_split', 2),
            min_samples_leaf=self.hyperparameters.get('min_samples_leaf', 1),
            max_features=self.hyperparameters.get('max_features', 'sqrt'),
            random_state=self.hyperparameters.get('random_state', 42),
            n_jobs=self.hyperparameters.get('n_jobs', -1),
            class_weight=self.hyperparameters.get('class_weight', 'balanced'),
            verbose=self.hyperparameters.get('verbose', 0)
        )
        return MultiOutputClassifier(rf, n_jobs=-1)
