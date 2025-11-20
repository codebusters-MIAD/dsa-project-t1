-- FilmLens Database Schema - Update Movies Catalog Schema
-- Version: V8
-- Description: Update movies_catalog table to support new data structure with IMDB/TMDB IDs and array types

-- Paso 1: Limpiar todos los datos existentes
TRUNCATE TABLE movies_catalog RESTART IDENTITY CASCADE;

-- Paso 2: Eliminar constraint UNIQUE de movie_id antes de eliminar índices
ALTER TABLE movies_catalog DROP CONSTRAINT IF EXISTS movies_catalog_movie_id_key;

-- Paso 2b: Eliminar todos los índices existentes (excepto PRIMARY KEY)
DROP INDEX IF EXISTS idx_movies_catalog_movie_id;
DROP INDEX IF EXISTS idx_movies_catalog_rating;
DROP INDEX IF EXISTS idx_movies_catalog_year;
DROP INDEX IF EXISTS idx_movies_catalog_genre;
DROP INDEX IF EXISTS idx_movies_catalog_movie_name;

-- Paso 3: Agregar nuevas columnas
ALTER TABLE movies_catalog 
    ADD COLUMN IF NOT EXISTS imdb_id BIGINT,
    ADD COLUMN IF NOT EXISTS tmdb_id BIGINT;

ALTER TABLE movies_catalog 
    ADD COLUMN genre_array TEXT[],
    ADD COLUMN director_array TEXT[],
    ADD COLUMN star_array TEXT[];

ALTER TABLE movies_catalog 
    DROP COLUMN IF EXISTS movie_id,
    DROP COLUMN IF EXISTS votes,
    DROP COLUMN IF EXISTS genre,
    DROP COLUMN IF EXISTS director,
    DROP COLUMN IF EXISTS star;

ALTER TABLE movies_catalog 
    RENAME COLUMN genre_array TO genre;
ALTER TABLE movies_catalog 
    RENAME COLUMN director_array TO director;
ALTER TABLE movies_catalog 
    RENAME COLUMN star_array TO star;

ALTER TABLE movies_catalog 
    ADD COLUMN runtime_int SMALLINT;

ALTER TABLE movies_catalog 
    DROP COLUMN runtime;

ALTER TABLE movies_catalog 
    RENAME COLUMN runtime_int TO runtime;

ALTER TABLE movies_catalog 
    ALTER COLUMN rating TYPE REAL USING rating::REAL;

-- Paso 4: Agregar restricciones de validación
ALTER TABLE movies_catalog 
    ADD CONSTRAINT chk_runtime_positive CHECK (runtime > 0 and runtime < 500),
    ADD CONSTRAINT chk_rating_range CHECK (rating >= 0 AND rating <= 10),
    ADD CONSTRAINT chk_year_valid CHECK (year >= 1800 AND year <= 2100);

-- Paso 5: Crear nuevos índices para el esquema actualizado
-- Índice único para TMDB ID
CREATE UNIQUE INDEX idx_movies_catalog_tmdb_id ON movies_catalog(tmdb_id) WHERE tmdb_id IS NOT NULL;

-- Índice para IMDB ID
CREATE INDEX idx_movies_catalog_imdb_id ON movies_catalog(imdb_id) WHERE imdb_id IS NOT NULL;

-- Índices GIN para búsquedas en arrays
CREATE INDEX idx_movies_catalog_genre_gin ON movies_catalog USING gin(genre);
CREATE INDEX idx_movies_catalog_director_gin ON movies_catalog USING gin(director);
CREATE INDEX idx_movies_catalog_star_gin ON movies_catalog USING gin(star);

-- Índices simples para columnas comunes de búsqueda/ordenamiento
CREATE INDEX idx_movies_catalog_rating ON movies_catalog(rating);
CREATE INDEX idx_movies_catalog_year ON movies_catalog(year);
CREATE INDEX idx_movies_catalog_movie_name ON movies_catalog(movie_name);
