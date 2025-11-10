-- FilmLens Database Schema - Movies Catalog
-- Version: V5

-- Table: movies_catalog
-- Stores movie information including IMDB data
CREATE TABLE IF NOT EXISTS movies_catalog (
    id SERIAL PRIMARY KEY,
    movie_id VARCHAR(50) NOT NULL UNIQUE,
    movie_name TEXT NOT NULL,
    year INTEGER,
    certificate VARCHAR(20),
    runtime VARCHAR(20),
    genre TEXT,
    rating DECIMAL(3,1),
    description TEXT,
    director TEXT,
    director_id VARCHAR(100),
    star TEXT,
    star_id TEXT,
    votes INTEGER,
    gross_in_usd NUMERIC(15,2),    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast lookups and queries
CREATE INDEX idx_movies_catalog_movie_id ON movies_catalog(movie_id);
CREATE INDEX idx_movies_catalog_rating ON movies_catalog(rating DESC);
CREATE INDEX idx_movies_catalog_year ON movies_catalog(year);
CREATE INDEX idx_movies_catalog_genre ON movies_catalog USING gin(to_tsvector('english', genre));
CREATE INDEX idx_movies_catalog_movie_name ON movies_catalog USING gin(to_tsvector('english', movie_name));
