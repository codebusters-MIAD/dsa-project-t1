"""Prediction module for multi-label classifier."""

import pandas as pd
from typing import Union, Dict, List
import logging
import numpy as np

from filmlens.processing.data_manager import load_pipeline
from filmlens.config import config

logger = logging.getLogger(__name__)


def make_prediction(input_data: Union[pd.DataFrame, dict, List[dict]]) -> dict:
    """
    Make predictions using saved pipeline.
    
    Args:
        input_data: DataFrame, dict, or list of dicts with input features
        
    Returns:
        Dictionary with predictions and probabilities per trigger
    """
    # Load trained pipeline
    pipeline = load_pipeline()
    
    # Convert to DataFrame if needed
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        input_data = pd.DataFrame(input_data)
    
    # Ensure text column exists
    text_col = config.data_config.text_column
    if text_col not in input_data.columns:
        raise ValueError(f"Missing required column: {text_col}")
    
    # Make predictions
    predictions = pipeline.predict(input_data)
    
    # Get probabilities if available
    try:
        probabilities = pipeline.predict_proba(input_data)
    except AttributeError:
        probabilities = None
    
    # Format results
    target_cols = config.data_config.target_columns
    results = []
    
    for i in range(len(input_data)):
        pred_dict = {
            'predictions': {}
        }
        
        for j, trigger_name in enumerate(target_cols):
            pred_dict['predictions'][trigger_name] = {
                'detected': bool(predictions[i, j]),
                'confidence': float(probabilities[i][j][1]) if probabilities else None
            }
        
        results.append(pred_dict)
    
    logger.info(f"Made {len(results)} predictions")
    
    return {
        'results': results,
        'version': config.app_config.package_name
    }
