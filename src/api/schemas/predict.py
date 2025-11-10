from typing import List
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for movie trigger prediction."""
    
    movie_id: str = Field(..., min_length=1, max_length=50, description="Unique movie identifier")
    title: str = Field(..., min_length=1, max_length=500, description="Movie title")
    description: str = Field(..., min_length=10, description="Movie plot or description")
    genre: str = Field(..., description="Primary movie genre")
    verbose: bool = Field(default=True, description="If False, returns only status 200 with 'OK'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "movie_id": "tt0468569",
                "title": "The Dark Knight",
                "description": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
                "genre": "Action",
                "verbose": True
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
    
    movie_id: str
    movie_title: str
    predictions: List[TriggerPrediction]
    
    class Config:
        json_schema_extra = {
            "example": {
                "movie_id": "tt0468569",
                "movie_title": "The Dark Knight",
                "predictions": [
                    {"trigger": "has_violence", "probability": 0.85, "detected": True},
                    {"trigger": "has_sexual_content", "probability": 0.12, "detected": False},
                    {"trigger": "has_substance_abuse", "probability": 0.23, "detected": False}
                ]
            }
        }
