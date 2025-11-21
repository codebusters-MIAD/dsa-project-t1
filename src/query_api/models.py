from sqlalchemy import Column, Integer, String, Float, TIMESTAMP, Text, SmallInteger, ARRAY
from sqlalchemy.sql import func

from .database import Base


class MovieTrigger(Base):
    """Movie triggers detection results (V7 multilevel schema)."""
    
    __tablename__ = "movie_triggers"
    
    id = Column(Integer, primary_key=True, index=True)
    movie_id = Column(String(50), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=False)
    description = Column(Text)
    detected_at = Column(TIMESTAMP, server_default=func.current_timestamp(), index=True)
    
    # Violencia
    violencia_nivel = Column(String(20))
    violencia_probabilidad = Column(Float)
    violencia_prob_sin_contenido = Column(Float)
    violencia_prob_moderado = Column(Float)
    violencia_prob_alto = Column(Float)
    
    # Sexualidad
    sexualidad_nivel = Column(String(20))
    sexualidad_probabilidad = Column(Float)
    sexualidad_prob_sin_contenido = Column(Float)
    sexualidad_prob_moderado = Column(Float)
    sexualidad_prob_alto = Column(Float)
    
    # Drogas
    drogas_nivel = Column(String(20))
    drogas_probabilidad = Column(Float)
    drogas_prob_sin_contenido = Column(Float)
    drogas_prob_moderado = Column(Float)
    drogas_prob_alto = Column(Float)
    
    # Lenguaje fuerte
    lenguaje_fuerte_nivel = Column(String(20))
    lenguaje_fuerte_probabilidad = Column(Float)
    lenguaje_fuerte_prob_sin_contenido = Column(Float)
    lenguaje_fuerte_prob_moderado = Column(Float)
    lenguaje_fuerte_prob_alto = Column(Float)
    
    # Suicidio
    suicidio_nivel = Column(String(20))
    suicidio_probabilidad = Column(Float)
    suicidio_prob_sin_contenido = Column(Float)
    suicidio_prob_moderado = Column(Float)
    suicidio_prob_alto = Column(Float)
    
    # Metadata
    model_version = Column(String(50))
    processing_time_ms = Column(Integer)


class MoviesCatalog(Base):
    """Movies catalog table."""
    
    __tablename__ = "movies_catalog"
    
    id = Column(Integer, primary_key=True, index=True)
    imdb_id = Column(String(50), unique=True, index=True)
    tmdb_id = Column(String(50), unique=True, index=True)
    movie_name = Column(Text, nullable=False, index=True)
    year = Column(Integer, index=True)
    runtime = Column(SmallInteger)
    genre = Column(ARRAY(Text), index=True)
    rating = Column(Float, index=True)
    description = Column(Text)
    director = Column(ARRAY(Text))
    star = Column(ARRAY(Text))
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())
