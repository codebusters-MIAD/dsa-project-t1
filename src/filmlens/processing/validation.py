"""Funciones de validacion y evaluacion de modelos."""

import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


def evaluate_multilabel_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_labels: List[str]
) -> Dict[str, float]:
    """
    Evaluar modelo multi-label con metricas completas.
    
    Args:
        y_true: Labels verdaderos (n_samples, n_labels)
        y_pred: Predicciones (n_samples, n_labels)
        target_labels: Nombres de los labels
        
    Returns:
        Diccionario con metricas globales y por label
    """
    metrics = {}
    
    # Metricas globales
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    metrics['subset_accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Metricas por label
    f1_per_label = f1_score(y_true, y_pred, average=None, zero_division=0)
    precision_per_label = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_label = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, label in enumerate(target_labels):
        metrics[f'f1_{label}'] = f1_per_label[i]
        metrics[f'precision_{label}'] = precision_per_label[i]
        metrics[f'recall_{label}'] = recall_per_label[i]
    
    return metrics
