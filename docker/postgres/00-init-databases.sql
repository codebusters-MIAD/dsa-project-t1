-- Inicialización de Bases de Datos para FilmLens

-- Base de datos para la aplicación FilmLens 
CREATE DATABASE mlflow
    WITH 
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.utf8'
    LC_CTYPE = 'en_US.utf8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;

COMMENT ON DATABASE mlflow IS 'MLflow Tracking Server - Experimentos y Artefactos';

-- Crear usuario para MLflow (con acceso solo a su DB)
CREATE USER mlflow_user WITH PASSWORD 'mlflow_dev_2025';
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow_user;

-- Conectar a la DB de mlflow para configurar permisos
\c mlflow

-- Dar permisos completos al usuario mlflow sobre el schema public
GRANT ALL ON SCHEMA public TO mlflow_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO mlflow_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO mlflow_user;


\c filmlens

