"""Router for movie_triggers table queries."""

import logging
import math
from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from ..database import get_db
from ..models import MovieTrigger
from ..schemas import PaginatedTriggersResponse, MovieTriggerResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/triggers", tags=["Movie Triggers"])


@router.get("", response_model=PaginatedTriggersResponse)
async def get_movie_triggers(
    page: int = Query(1, ge=1, description="Page number (starts at 1)"),
    limit: int = Query(10, ge=1, le=100, description="Results per page (max 100)"),
    movie_id: Optional[str] = Query(None, description="Filter by movie_id"),
    has_violence: Optional[bool] = Query(None, description="Filter by violence trigger"),
    has_sexual_content: Optional[bool] = Query(None, description="Filter by sexual content trigger"),
    has_substance_abuse: Optional[bool] = Query(None, description="Filter by substance abuse trigger"),
    has_suicide: Optional[bool] = Query(None, description="Filter by suicide trigger"),
    has_strong_language: Optional[bool] = Query(None, description="Filter by strong language trigger"),
    db: Session = Depends(get_db)
):
    """
    Get paginated list of movie triggers.
    
    Query Parameters:
    - page: Page number (default: 1)
    - limit: Results per page (default: 10, max: 100)
    - movie_id: Filter by specific movie ID
    - has_violence: Filter by violence trigger (true/false)
    - has_sexual_content: Filter by sexual content trigger (true/false)
    - has_substance_abuse: Filter by substance abuse trigger (true/false)
    - has_suicide: Filter by suicide trigger (true/false)
    - has_strong_language: Filter by strong language trigger (true/false)
    
    Returns:
    - Paginated list of movie trigger records with movie_id as first field
    """
    try:
        # Build query with filters
        query = db.query(MovieTrigger)
        
        # Apply filters
        if movie_id is not None:
            query = query.filter(MovieTrigger.movie_id == movie_id)
        
        if has_violence is not None:
            query = query.filter(MovieTrigger.has_violence == has_violence)
        
        if has_sexual_content is not None:
            query = query.filter(MovieTrigger.has_sexual_content == has_sexual_content)
        
        if has_substance_abuse is not None:
            query = query.filter(MovieTrigger.has_substance_abuse == has_substance_abuse)
        
        if has_suicide is not None:
            query = query.filter(MovieTrigger.has_suicide == has_suicide)
        
        if has_strong_language is not None:
            query = query.filter(MovieTrigger.has_strong_language == has_strong_language)
        
        # Get total count
        total_items = query.count()
        
        # Calculate pagination
        total_pages = math.ceil(total_items / limit) if total_items > 0 else 0
        offset = (page - 1) * limit
        
        # Get paginated results (order by indexed column for performance)
        items = query.order_by(MovieTrigger.detected_at).offset(offset).limit(limit).all()
        
        # Convert to response models
        movie_triggers = [MovieTriggerResponse.model_validate(item) for item in items]
        
        return PaginatedTriggersResponse(
            page=page,
            limit=limit,
            total_items=total_items,
            total_pages=total_pages,
            items=movie_triggers
        )
        
    except SQLAlchemyError as e:
        logger.error(f"Error querying movie triggers: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while querying the database."
        )


@router.get("/{movie_id}", response_model=MovieTriggerResponse)
async def get_movie_trigger_by_id(
    movie_id: str,
    db: Session = Depends(get_db)
):
    """
    Get a specific movie trigger by movie_id.
    
    Parameters:
    - movie_id: The IMDb movie ID (e.g., tt0111161)
    
    Returns:
    - Movie trigger record with movie_id as first field
    """
    try:
        trigger = db.query(MovieTrigger).filter(
            MovieTrigger.movie_id == movie_id
        ).first()
        
        if not trigger:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Movie trigger not found for movie_id: {movie_id}"
            )
        
        return MovieTriggerResponse.model_validate(trigger)
        
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Error getting movie trigger {movie_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while querying the database."
        )
