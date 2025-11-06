"""Health check endpoints."""

import logging
from fastapi import APIRouter

from ..schemas.health import HealthResponse
from ..config import settings
from ..models import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the API health status and model availability.
    """
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "unhealthy",
        model_loaded=model_manager.is_loaded,
        version=settings.version
    )


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "description": settings.description,
        "endpoints": {
            "health": "/health",
            "predict": "/api/v1/predict",
            "docs": "/docs"
        }
    }
