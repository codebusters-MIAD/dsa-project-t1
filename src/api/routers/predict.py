import logging
import time
import pandas as pd
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from ..schemas.predict import PredictionRequest, PredictionResponse, TriggerPrediction
from ..models import model_manager
from ..database import db_manager
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Prediction"])

TRIGGER_LABELS = [
    'has_suicide',
    'has_substance_abuse',
    'has_strong_language',
    'has_sexual_content',
    'has_violence'
]


@router.post("/predict")
async def predict_triggers(request: PredictionRequest):
    """
    Predict sensitive content triggers for a movie and save to database.
    
    Args:
        request: Movie information (movie_id, title, description, genre, verbose)
        
    Returns:
        Multi-label predictions for 5 trigger categories (if verbose=True)
        or simple {"status": "OK"} (if verbose=False)
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        start_time = time.time()
        
        # Ensure genre is string type (not object)
        input_data = pd.DataFrame([{
            'description': request.description,
            'genre': str(request.genre)
        }])
        
        # Ensure correct dtypes
        input_data['description'] = input_data['description'].astype(str)
        input_data['genre'] = input_data['genre'].astype(str)
        
        predictions_array = model_manager.predict(input_data)
        
        predictions = []
        
        try:
            proba_list = model_manager.predict_proba(input_data)
        except (AttributeError, ValueError, IndexError) as e:
            logger.warning(f"Could not get probabilities: {e}")
            proba_list = None
        
        db_predictions = {}
        
        for i, trigger_label in enumerate(TRIGGER_LABELS):
            detected = bool(predictions_array[0][i])
            
            if proba_list is not None:
                prob = float(proba_list[i][0, 1])
            else:
                prob = 1.0 if detected else 0.0
            
            predictions.append(
                TriggerPrediction(
                    trigger=trigger_label,
                    probability=prob,
                    detected=detected
                )
            )
            
            db_predictions[trigger_label] = detected
            db_predictions[f"{trigger_label.replace('has_', '')}_confidence"] = prob
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Save to database
        db_manager.save_prediction(
            movie_id=request.movie_id,
            title=request.title,
            description=request.description,
            predictions=db_predictions,
            model_version=settings.model_file,
            processing_time_ms=processing_time_ms
        )
        
        # Return based on verbose flag
        if not request.verbose:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"status": "OK"}
            )
        
        return PredictionResponse(
            movie_id=request.movie_id,
            movie_title=request.title,
            predictions=predictions
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/predict/{movie_id}", status_code=status.HTTP_200_OK)
async def get_prediction(movie_id: str):
    """
    Retrieve saved prediction for a movie from database.
    
    Args:
        movie_id: Unique movie identifier
        
    Returns:
        Saved prediction data if found
    """
    try:
        result = db_manager.get_prediction(movie_id)
        
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No prediction found for movie_id: {movie_id}"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve prediction: {str(e)}"
        )
