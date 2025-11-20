-- FilmLens Database Schema - Fix Movie IDs to VARCHAR
-- Version: V9
-- Description: Change imdb_id and tmdb_id from BIGINT to VARCHAR(50), make both UNIQUE but not required

DROP INDEX IF EXISTS idx_movies_catalog_tmdb_id;
DROP INDEX IF EXISTS idx_movies_catalog_imdb_id;

ALTER TABLE movies_catalog 
    ALTER COLUMN imdb_id TYPE VARCHAR(50) USING imdb_id::VARCHAR(50);

ALTER TABLE movies_catalog 
    ALTER COLUMN tmdb_id TYPE VARCHAR(50) USING tmdb_id::VARCHAR(50);

CREATE UNIQUE INDEX idx_movies_catalog_imdb_id ON movies_catalog(imdb_id) WHERE imdb_id IS NOT NULL;
CREATE UNIQUE INDEX idx_movies_catalog_tmdb_id ON movies_catalog(tmdb_id) WHERE tmdb_id IS NOT NULL;

COMMENT ON COLUMN movies_catalog.imdb_id IS 'IMDB ID (e.g., tt1234567) - VARCHAR(50), UNIQUE but optional';
COMMENT ON COLUMN movies_catalog.tmdb_id IS 'The Movie Database (TMDB) ID - VARCHAR(50), UNIQUE but optional';
