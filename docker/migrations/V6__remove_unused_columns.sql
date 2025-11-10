-- FilmLens Database Schema - Remove unused columns from movies_catalog
-- Version: V6

-- Remove columns that are not needed
ALTER TABLE movies_catalog DROP COLUMN IF EXISTS gross_in_usd;
ALTER TABLE movies_catalog DROP COLUMN IF EXISTS director_id;
ALTER TABLE movies_catalog DROP COLUMN IF EXISTS certificate;
ALTER TABLE movies_catalog DROP COLUMN IF EXISTS star_id;
