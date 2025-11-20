from typing import List, Dict
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model para clasificacion de sensibilidad"""
    
    movie_id: str = Field(..., min_length=1, max_length=50, description="Identificador unico de pelicula")
    title: str = Field(..., min_length=1, max_length=500, description="Titulo de la pelicula")
    description: str = Field(..., min_length=10, description="Sinopsis o descripcion de la pelicula")
    
    class Config:
        json_schema_extra = {
            "example": {
                "movie_id": "tt0468569",
                "title": "The Dark Knight",
                "description": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."
            }
        }


class CategoryPrediction(BaseModel):
    """Prediccion de nivel de sensibilidad para una categoria"""
    
    nivel: str = Field(..., description="Nivel predicho: sin_contenido, moderado, alto")
    probabilidad: float = Field(..., ge=0.0, le=1.0, description="Probabilidad del nivel predicho")
    probabilidades_todas: Dict[str, float] = Field(..., description="Probabilidades de todos los niveles")
    
    class Config:
        json_schema_extra = {
            "example": {
                "nivel": "alto",
                "probabilidad": 0.85,
                "probabilidades_todas": {
                    "sin_contenido": 0.05,
                    "moderado": 0.10,
                    "alto": 0.85
                }
            }
        }


class PredictionResponse(BaseModel):
    """Response model para clasificacion de sensibilidad"""
    
    movie_id: str
    movie_title: str
    violencia_nivel: CategoryPrediction
    sexualidad_nivel: CategoryPrediction
    drogas_nivel: CategoryPrediction
    lenguaje_fuerte_nivel: CategoryPrediction
    suicidio_nivel: CategoryPrediction
    
    class Config:
        json_schema_extra = {
            "example": {
                "movie_id": "tt0468569",
                "movie_title": "The Dark Knight",
                "violencia_nivel": {
                    "nivel": "alto",
                    "probabilidad": 0.85,
                    "probabilidades_todas": {"sin_contenido": 0.05, "moderado": 0.10, "alto": 0.85}
                },
                "sexualidad_nivel": {
                    "nivel": "sin_contenido",
                    "probabilidad": 0.90,
                    "probabilidades_todas": {"sin_contenido": 0.90, "moderado": 0.08, "alto": 0.02}
                },
                "drogas_nivel": {
                    "nivel": "moderado",
                    "probabilidad": 0.55,
                    "probabilidades_todas": {"sin_contenido": 0.30, "moderado": 0.55, "alto": 0.15}
                },
                "lenguaje_fuerte_nivel": {
                    "nivel": "moderado",
                    "probabilidad": 0.65,
                    "probabilidades_todas": {"sin_contenido": 0.20, "moderado": 0.65, "alto": 0.15}
                },
                "suicidio_nivel": {
                    "nivel": "sin_contenido",
                    "probabilidad": 0.95,
                    "probabilidades_todas": {"sin_contenido": 0.95, "moderado": 0.03, "alto": 0.02}
                }
            }
        }
