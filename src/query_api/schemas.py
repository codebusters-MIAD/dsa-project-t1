from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class MovieTriggerResponse(BaseModel):    
    
    model_config = {"protected_namespaces": (), "from_attributes": True}
    
    movie_id: str
    id: int
    title: str
    description: Optional[str] = None
    detected_at: datetime
    
    # Violencia
    violencia_nivel: Optional[str] = None
    violencia_probabilidad: Optional[float] = None
    violencia_prob_sin_contenido: Optional[float] = None
    violencia_prob_moderado: Optional[float] = None
    violencia_prob_alto: Optional[float] = None
    
    # Sexualidad
    sexualidad_nivel: Optional[str] = None
    sexualidad_probabilidad: Optional[float] = None
    sexualidad_prob_sin_contenido: Optional[float] = None
    sexualidad_prob_moderado: Optional[float] = None
    sexualidad_prob_alto: Optional[float] = None
    
    # Drogas
    drogas_nivel: Optional[str] = None
    drogas_probabilidad: Optional[float] = None
    drogas_prob_sin_contenido: Optional[float] = None
    drogas_prob_moderado: Optional[float] = None
    drogas_prob_alto: Optional[float] = None
    
    # Lenguaje fuerte
    lenguaje_fuerte_nivel: Optional[str] = None
    lenguaje_fuerte_probabilidad: Optional[float] = None
    lenguaje_fuerte_prob_sin_contenido: Optional[float] = None
    lenguaje_fuerte_prob_moderado: Optional[float] = None
    lenguaje_fuerte_prob_alto: Optional[float] = None
    
    # Suicidio
    suicidio_nivel: Optional[str] = None
    suicidio_probabilidad: Optional[float] = None
    suicidio_prob_sin_contenido: Optional[float] = None
    suicidio_prob_moderado: Optional[float] = None
    suicidio_prob_alto: Optional[float] = None
    
    # Metadata
    model_version: Optional[str] = None
    processing_time_ms: Optional[int] = None


class MovieResponse(BaseModel):
    """Response for movies catalog."""
    
    model_config = {"from_attributes": True}
    
    imdb_id: Optional[str] = None
    tmdb_id: Optional[str] = None
    movie_name: str
    year: Optional[int] = None
    runtime: Optional[int] = None
    genre: Optional[List[str]] = None
    rating: Optional[float] = None
    description: Optional[str] = None
    director: Optional[List[str]] = None
    star: Optional[List[str]] = None


class AutocompleteItem(BaseModel):
    """Autocomplete suggestion."""
    
    imdb_id: Optional[str] = None
    movie_name: str
    year: Optional[int] = None


class AutocompleteResponse(BaseModel):
    """Autocomplete suggestions response."""
    
    suggestions: List[AutocompleteItem]


class FilterOptions(BaseModel):
    """Available filter options."""
    
    genres: List[str]
    year_range: dict
    rating_range: dict
    runtime_range: dict


class PaginatedMoviesResponse(BaseModel):
    """Paginated movies response."""
    
    page: int = Field(..., ge=1)
    limit: int = Field(..., ge=1, le=100)
    total_items: int = Field(..., ge=0)
    total_pages: int = Field(..., ge=0)
    items: List[MovieResponse]


class PaginatedTriggersResponse(BaseModel):    
    
    page: int = Field(..., ge=1, description="Current page number")
    limit: int = Field(..., ge=1, le=100, description="Results per page")
    total_items: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    items: List[MovieTriggerResponse] = Field(..., description="Movie trigger records")
