from sqlalchemy import Column, Integer, String, Boolean, Float, TIMESTAMP, Text
from sqlalchemy.sql import func

from .database import Base


class MovieTrigger(Base):
    """Movie triggers detection results."""
    
    __tablename__ = "movie_triggers"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Movie identification - movie_id always first column after id
    movie_id = Column(String(50), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=False)
    description = Column(Text)
    detected_at = Column(TIMESTAMP, server_default=func.current_timestamp(), index=True)
    
    # Multi-label triggers (boolean flags)
    has_suicide = Column(Boolean, default=False)
    has_substance_abuse = Column(Boolean, default=False)
    has_strong_language = Column(Boolean, default=False)
    has_sexual_content = Column(Boolean, default=False)
    has_violence = Column(Boolean, default=False, index=True)
    
    # Confidence scores (0-1)
    suicide_confidence = Column(Float)
    substance_abuse_confidence = Column(Float)
    strong_language_confidence = Column(Float)
    sexual_content_confidence = Column(Float)
    violence_confidence = Column(Float)
    
    # Metadata
    model_version = Column(String(50))
    processing_time_ms = Column(Integer)
