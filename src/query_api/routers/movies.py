"""Router for movies_catalog queries."""

import logging
import math
from typing import Optional, List
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func, or_, and_

from ..database import get_db
from ..models import MoviesCatalog
from ..schemas import (
    MovieResponse,
    PaginatedMoviesResponse,
    AutocompleteResponse,
    AutocompleteItem,
    FilterOptions
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/movies", tags=["Movies Catalog"])


@router.get("/search", response_model=PaginatedMoviesResponse)
async def search_movies(
    q: Optional[str] = Query(None, description="Search by movie name or description"),
    genres: Optional[List[str]] = Query(None, description="Filter by genres (multiple)"),
    year_from: Optional[int] = Query(None, ge=1800, le=2100, description="Year from"),
    year_to: Optional[int] = Query(None, ge=1800, le=2100, description="Year to"),
    runtime_min: Optional[int] = Query(None, ge=0, le=500, description="Min runtime in minutes"),
    runtime_max: Optional[int] = Query(None, ge=0, le=500, description="Max runtime in minutes"),
    rating_min: Optional[float] = Query(None, ge=0.0, le=10.0, description="Min rating"),
    rating_max: Optional[float] = Query(None, ge=0.0, le=10.0, description="Max rating"),
    sort_by: str = Query("year", enum=["year", "rating", "runtime", "movie_name"], description="Sort field"),
    order: str = Query("desc", enum=["asc", "desc"], description="Sort order"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    db: Session = Depends(get_db)
):
    """
    Search movies with dynamic filters.
    
    Parameters:
    - q: Text search in movie_name and description
    - genres: Filter by one or more genres
    - year_from/year_to: Year range filter
    - runtime_min/runtime_max: Runtime range filter
    - rating_min/rating_max: Rating range filter
    - sort_by: Field to sort by
    - order: Sort order (asc/desc)
    - page: Page number (starts at 1)
    - limit: Results per page (max 100)
    """
    try:
        query = db.query(MoviesCatalog)
        
        # Text search
        if q:
            search_term = f"%{q}%"
            query = query.filter(
                or_(
                    MoviesCatalog.movie_name.ilike(search_term),
                    MoviesCatalog.description.ilike(search_term)
                )
            )
        
        # Genre filter (array overlap using &&)
        if genres:
            query = query.filter(MoviesCatalog.genre.op('&&')(genres))
        
        # Year range
        if year_from:
            query = query.filter(MoviesCatalog.year >= year_from)
        if year_to:
            query = query.filter(MoviesCatalog.year <= year_to)
        
        # Runtime range
        if runtime_min:
            query = query.filter(MoviesCatalog.runtime >= runtime_min)
        if runtime_max:
            query = query.filter(MoviesCatalog.runtime <= runtime_max)
        
        # Rating range
        if rating_min:
            query = query.filter(MoviesCatalog.rating >= rating_min)
        if rating_max:
            query = query.filter(MoviesCatalog.rating <= rating_max)
        
        # Count total
        total_items = query.count()
        total_pages = math.ceil(total_items / limit) if total_items > 0 else 0
        
        # Sorting
        sort_column = getattr(MoviesCatalog, sort_by)
        if order == "desc":
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())
        
        # Pagination
        offset = (page - 1) * limit
        items = query.offset(offset).limit(limit).all()
        
        movies = [MovieResponse.model_validate(item) for item in items]
        
        return PaginatedMoviesResponse(
            page=page,
            limit=limit,
            total_items=total_items,
            total_pages=total_pages,
            items=movies
        )
        
    except SQLAlchemyError as e:
        logger.error(f"Error searching movies: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error querying database"
        )


@router.get("/autocomplete", response_model=AutocompleteResponse)
async def autocomplete_movies(
    q: str = Query(..., min_length=2, description="Search term (min 2 chars)"),
    limit: int = Query(10, ge=1, le=20, description="Max suggestions"),
    db: Session = Depends(get_db)
):
    """
    Fast autocomplete for movie names.
    
    Parameters:
    - q: Search term (minimum 2 characters)
    - limit: Maximum number of suggestions (default 10, max 20)
    
    Returns:
    - List of movie suggestions with imdb_id, movie_name, year
    """
    try:
        search_term = f"%{q}%"
        
        results = db.query(
            MoviesCatalog.imdb_id,
            MoviesCatalog.movie_name,
            MoviesCatalog.year
        ).filter(
            MoviesCatalog.movie_name.ilike(search_term)
        ).order_by(
            MoviesCatalog.rating.desc().nulls_last()
        ).limit(limit).all()
        
        suggestions = [
            AutocompleteItem(
                imdb_id=r.imdb_id,
                movie_name=r.movie_name,
                year=r.year
            )
            for r in results
        ]
        
        return AutocompleteResponse(suggestions=suggestions)
        
    except SQLAlchemyError as e:
        logger.error(f"Error in autocomplete: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error querying database"
        )


@router.get("/filters/options", response_model=FilterOptions)
async def get_filter_options(db: Session = Depends(get_db)):
    """
    Get available filter options.
    
    Returns:
    - genres: List of unique genres
    - year_range: {min, max}
    - rating_range: {min, max}
    - runtime_range: {min, max}
    """
    try:
        # Get unique genres
        genres_result = db.query(
            func.unnest(MoviesCatalog.genre).label('genre')
        ).distinct().all()
        genres = sorted([g.genre for g in genres_result if g.genre])
        
        # Get year range
        year_stats = db.query(
            func.min(MoviesCatalog.year).label('min'),
            func.max(MoviesCatalog.year).label('max')
        ).first()
        
        # Get rating range
        rating_stats = db.query(
            func.min(MoviesCatalog.rating).label('min'),
            func.max(MoviesCatalog.rating).label('max')
        ).first()
        
        # Get runtime range
        runtime_stats = db.query(
            func.min(MoviesCatalog.runtime).label('min'),
            func.max(MoviesCatalog.runtime).label('max')
        ).first()
        
        return FilterOptions(
            genres=genres,
            year_range={
                "min": year_stats.min if year_stats.min else 1920,
                "max": year_stats.max if year_stats.max else 2025
            },
            rating_range={
                "min": float(rating_stats.min) if rating_stats.min else 0.0,
                "max": float(rating_stats.max) if rating_stats.max else 10.0
            },
            runtime_range={
                "min": runtime_stats.min if runtime_stats.min else 0,
                "max": runtime_stats.max if runtime_stats.max else 300
            }
        )
        
    except SQLAlchemyError as e:
        logger.error(f"Error getting filter options: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error querying database"
        )


@router.get("/{movie_id}", response_model=MovieResponse)
async def get_movie_by_id(
    movie_id: str,
    db: Session = Depends(get_db)
):
    """
    Get movie details by IMDB ID or TMDB ID.
    
    Parameters:
    - movie_id: IMDB ID (e.g., tt0468569) or TMDB ID
    
    Returns:
    - Complete movie information
    """
    try:
        movie = db.query(MoviesCatalog).filter(
            or_(
                MoviesCatalog.imdb_id == movie_id,
                MoviesCatalog.tmdb_id == movie_id
            )
        ).first()
        
        if not movie:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Movie not found with id: {movie_id}"
            )
        
        return MovieResponse.model_validate(movie)
        
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Error getting movie {movie_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error querying database"
        )
