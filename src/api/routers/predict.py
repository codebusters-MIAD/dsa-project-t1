import logging
import time
from fastapi import APIRouter, HTTPException, status

from ..schemas.predict import PredictionRequest, PredictionResponse, CategoryPrediction
from ..models import model_manager
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_sensitivity(request: PredictionRequest):
    """
    Clasifica niveles de sensibilidad para una pelicula.
    
    Args:
        request: Informacion de la pelicula (movie_id, title, description)
        
    Returns:
        Predicciones multiclase para 5 categorias de sensibilidad
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no cargado. Revisar logs del servidor."
        )
    
    try:
        start_time = time.time()
        
        # Obtener predicciones del modelo
        predictions = model_manager.predict(request.description)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Prediccion completada en {processing_time_ms}ms para movie_id={request.movie_id}")
        
        # Construir response
        response = PredictionResponse(
            movie_id=request.movie_id,
            movie_title=request.title,
            violencia_nivel=CategoryPrediction(**predictions['violencia_nivel']),
            sexualidad_nivel=CategoryPrediction(**predictions['sexualidad_nivel']),
            drogas_nivel=CategoryPrediction(**predictions['drogas_nivel']),
            lenguaje_fuerte_nivel=CategoryPrediction(**predictions['lenguaje_fuerte_nivel']),
            suicidio_nivel=CategoryPrediction(**predictions['suicidio_nivel'])
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error en prediccion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fallo en prediccion: {str(e)}"
        )



