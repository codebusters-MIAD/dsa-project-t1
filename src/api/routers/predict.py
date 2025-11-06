import logging
import pandas as pd
from fastapi import APIRouter, HTTPException, status

from ..schemas.predict import PredictionRequest, PredictionResponse, TriggerPrediction
from ..models import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Prediction"])

TRIGGER_LABELS = [
    'has_violence',
    'has_sexual_content',
    'has_substance_abuse',
    'has_suicide',
    'has_child_abuse',
    'has_discrimination',
    'has_strong_language',
    'has_horror',
    'has_animal_cruelty'
]


@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_triggers(request: PredictionRequest):
    """
    Predict sensitive content triggers for a movie.
    
    Args:
        request: Movie information (title, description, genre)
        
    Returns:
        Multi-label predictions for 9 trigger categories
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Prepare input data as DataFrame
        input_data = pd.DataFrame([{
            'description': request.description,
            'genre': request.genre
        }])
        
        # Get predictions
        predictions_array = model_manager.predict(input_data)
        
        # predictions_array shape: (1, 9) for multi-label
        # Each element is binary (0 or 1)
        predictions = []
        
        # Get probabilities for all outputs
        # predict_proba returns a list of arrays (one per output), not a 2D array
        try:
            proba_list = model_manager.predict_proba(input_data)
        except (AttributeError, ValueError, IndexError) as e:
            logger.warning(f"Could not get probabilities: {e}")
            proba_list = None
        
        for i, trigger_label in enumerate(TRIGGER_LABELS):
            detected = bool(predictions_array[0][i])
            
            # Extract probability for the positive class (index 1)
            if proba_list is not None:
                # proba_list[i] shape: (n_samples, 2)
                # [0, 1] gets first sample, positive class probability
                prob = float(proba_list[i][0, 1])
            else:
                # Fallback to binary value if proba not available
                prob = 1.0 if detected else 0.0
            
            predictions.append(
                TriggerPrediction(
                    trigger=trigger_label,
                    probability=prob,
                    detected=detected
                )
            )
        
        return PredictionResponse(
            movie_title=request.title,
            predictions=predictions
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
