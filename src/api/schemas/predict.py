from typing import List
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for movie trigger prediction."""
    
    title: str = Field(..., min_length=1, max_length=500, description="Movie title")
    description: str = Field(..., min_length=10, description="Movie plot or description")
    genre: str = Field(..., description="Primary movie genre")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "The Dark Knight",
                "description": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
                "genre": "Action"
            }
        }


class TriggerPrediction(BaseModel):
    """Individual trigger prediction result."""
    
    trigger: str = Field(..., description="Trigger category name")
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction probability (0-1)")
    detected: bool = Field(..., description="Whether trigger is detected (threshold > 0.5)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "trigger": "has_violence",
                "probability": 0.85,
                "detected": True
            }
        }


class PredictionResponse(BaseModel):
    """Response model for movie trigger prediction."""
    
    movie_title: str
    predictions: List[TriggerPrediction]
    
    class Config:
        json_schema_extra = {
            "example": {
                "movie_title": "The Dark Knight",
                "predictions": [
                    {"trigger": "has_violence", "probability": 0.85, "detected": True},
                    {"trigger": "has_sexual_content", "probability": 0.12, "detected": False},
                    {"trigger": "has_substance_abuse", "probability": 0.23, "detected": False}
                ]
            }
        }
