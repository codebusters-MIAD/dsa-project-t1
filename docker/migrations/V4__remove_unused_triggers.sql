-- FilmLens Database Schema - Remove Unused Triggers
-- Version: V4
-- Description: Remove triggers not used in 5-trigger model

-- Eliminar columnas de triggers no utilizados
ALTER TABLE movie_triggers 
    DROP COLUMN IF EXISTS has_child_abuse,
    DROP COLUMN IF EXISTS has_discrimination,
    DROP COLUMN IF EXISTS has_horror,
    DROP COLUMN IF EXISTS has_animal_cruelty;

-- Eliminar columnas de confidence scores no utilizados
ALTER TABLE movie_triggers 
    DROP COLUMN IF EXISTS child_abuse_confidence,
    DROP COLUMN IF EXISTS discrimination_confidence,
    DROP COLUMN IF EXISTS horror_confidence,
    DROP COLUMN IF EXISTS animal_cruelty_confidence;

-- Eliminar indices de triggers eliminados
DROP INDEX IF EXISTS idx_movie_triggers_horror;

-- Comentario sobre los 5 triggers activos
COMMENT ON TABLE movie_triggers IS 'Movie triggers detection - 5 active triggers: suicide, substance_abuse, strong_language, sexual_content, violence';
