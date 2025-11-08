from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from .base import BaseClassifier
from .registry import ModelRegistry


@ModelRegistry.register("logistic_regression")
class LogisticRegressionModel(BaseClassifier):
    """Clasificador Regresion Logistica multi-label."""
    
    @property
    def name(self) -> str:
        return "LogisticRegression"
    
    def build(self) -> MultiOutputClassifier:
        lr = LogisticRegression(
            C=self.hyperparameters.get('C', 1.0),
            penalty=self.hyperparameters.get('penalty', 'l2'),
            solver=self.hyperparameters.get('solver', 'lbfgs'),
            max_iter=self.hyperparameters.get('max_iter', 1000),
            random_state=self.hyperparameters.get('random_state', 42),
            n_jobs=self.hyperparameters.get('n_jobs', -1),
            class_weight=self.hyperparameters.get('class_weight', 'balanced'),
            verbose=self.hyperparameters.get('verbose', 0)
        )
        return MultiOutputClassifier(lr, n_jobs=-1)
