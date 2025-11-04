-- FilmLens Database Initialization
-- Creates application user and database
-- This script runs automatically when PostgreSQL container starts for the first time

-- Create application user if not exists
DO
$$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles WHERE rolname = 'filmlens_user'
   ) THEN
      CREATE USER filmlens_user WITH PASSWORD 'filmlens_dev_2025';
   END IF;
END
$$;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE filmlens TO filmlens_user;
ALTER DATABASE filmlens OWNER TO filmlens_user;

-- Grant schema privileges (will be executed after database creation)
\c filmlens
GRANT ALL ON SCHEMA public TO filmlens_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO filmlens_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO filmlens_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO filmlens_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO filmlens_user;
