from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class MovieTriggerResponse(BaseModel):    
    
    model_config = {"protected_namespaces": (), "from_attributes": True}
    
    # movie_id is always first
    movie_id: str
    id: int
    title: str
    description: Optional[str] = None
    detected_at: datetime
    
    # Trigger flags
    has_violence: bool
    has_sexual_content: bool
    has_substance_abuse: bool
    has_suicide: bool
    has_child_abuse: bool
    has_discrimination: bool
    has_strong_language: bool
    has_horror: bool
    has_animal_cruelty: bool
    
    # Confidence scores
    violence_confidence: Optional[float] = None
    sexual_content_confidence: Optional[float] = None
    substance_abuse_confidence: Optional[float] = None
    suicide_confidence: Optional[float] = None
    child_abuse_confidence: Optional[float] = None
    discrimination_confidence: Optional[float] = None
    strong_language_confidence: Optional[float] = None
    horror_confidence: Optional[float] = None
    animal_cruelty_confidence: Optional[float] = None
    
    # Metadata
    model_version: Optional[str] = None
    processing_time_ms: Optional[int] = None


class PaginatedResponse(BaseModel):    
    
    page: int = Field(..., ge=1, description="Current page number")
    limit: int = Field(..., ge=1, le=100, description="Results per page")
    total_items: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    items: list[MovieTriggerResponse] = Field(..., description="Movie trigger records")
