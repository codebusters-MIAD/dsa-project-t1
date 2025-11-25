# FilmLens - Guía Técnica del Proyecto

## Introducción

FilmLens es una plataforma de recomendación de películas con análisis de sensibilidad para contenido familiar. El proyecto incluye modelos de Machine Learning para detectar triggers sensibles en películas (violencia, lenguaje inapropiado, contenido adulto, etc.) y proporcionar puntuaciones de idoneidad familiar.

## Arquitectura General

El proyecto está dividido en varios componentes principales dentro de `src/`:

Usando docker compose, permite diseñar localmente una arquitectura basada en contenedores.

![Docker Compose](images/ArquitecturaFilmLens-FilmLensFinal.png)

### 1. **API de Predicción** (`src/api/`)

FastAPI service que expone endpoints para obtener predicciones de sensibilidad de películas.

**Características:**
- Predicciones en tiempo real usando modelos ML entrenados
- Conexión con PostgreSQL para almacenar resultados
- Integración con AWS Secrets Manager para credenciales

**Endpoints principales:**
- `GET /health` - Estado del servicio
- `POST /predict` - Predicción de sensibilidad de película

**Archivos clave:**
- `main.py` - Aplicación FastAPI principal
- `config.py` - Configuración del servicio
- `models.py` - Gestión del modelo ML
- `database.py` - Conexión a PostgreSQL

### 2. **Query API** (`src/query_api/`)

API de consulta para acceder a los datos históricos de predicciones y películas almacenadas.

**Características:**
- Consultas a base de datos PostgreSQL
- Endpoints para búsqueda de películas
- Historial de triggers detectados
- CRUD operations sobre catálogo de películas

**Endpoints principales:**
- `GET /health` - Estado del servicio
- `GET /movies/` - Lista de películas
- `GET /movies/{movie_id}` - Detalles de película
- `GET /triggers/` - Triggers detectados

**Archivos clave:**
- `main.py` - Aplicación FastAPI
- `routers/movies.py` - Endpoints de películas
- `routers/triggers.py` - Endpoints de triggers
- `schemas.py` - Modelos Pydantic

### 3. **Dashboard** (`src/dashboard/`)

Interfaz web interactiva construida con Dash/Plotly para explorar películas y sus puntuaciones de sensibilidad.

![Dashboard](images/dashboard.png)

**Características:**
- Búsqueda de películas con autocompletado
- Visualización de triggers por categoría
- Gauge de idoneidad familiar
- Filtros por edad y sensibilidades

**Estructura:**
- `app.py` - Aplicación Dash principal
- `components/` - Componentes de UI reutilizables
- `services/api_client.py` - Cliente para APIs
- `utils/` - Funciones auxiliares de cálculo y display

### 4. **FilmLens ML Pipeline ** (`src/filmlens/`)

Pipeline de entrenamiento para modelos multi-label de clasificación de triggers. Este pipeline de entrenamiento 
se inspiro en la estructura dada por el curso del proyecto bankchurn.

**Características:**
- Entrenamiento de modelos multi-label (múltiples triggers por película)
- Integración con MLflow para tracking
- Soporte para múltiples algoritmos (LightGBM, Random Forest, Logistic Regression)
- Feature engineering de keywords de películas

**Archivos clave:**
- `train_pipeline.py` - Pipeline de entrenamiento
- `predict.py` - Inferencia con modelo entrenado
- `models/` - Implementaciones de algoritmos
- `processing/` - Feature engineering y validación

### 5. **MovieLens ML Pipeline ** (`src/movielens/`)

Pipeline alternativo de entrenamiento con enfoque en sensibilidad multi-categoría. Este pipeline de entrenamiento 
se inspiro en la estructura dada por el curso del proyecto bankchurn

**Características:**
- Modelos de sensibilidad por múltiples categorías
- Evaluación detallada por categoría
- Integración con MLflow
- Feature engineering avanzado

**Archivos clave:**
- `train_pipeline.py` - Pipeline de entrenamiento
- `models/` - Registry de modelos
- `processing/` - Data managers y feature builders
- `config/config.yml` - Configuración del pipeline

## Requisitos Previos

```bash
# Python 3.9+
python --version

# Docker y Docker Compose
docker --version
docker-compose --version

# PostgreSQL 15 (para desarrollo local)
psql --version

# AWS CLI (para deployment)
aws --version
```

## Inicio Rápido

### 1. Configuración del Entorno

```bash
# Clonar repositorio
git clone <repository-url>
cd dsa-project-t1

# Crear archivo de variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# Instalar dependencias
pip install -r requirements/requirements.txt
```

### 2. Levantar Servicios con Docker

```bash
# Iniciar todos los servicios
docker-compose up -d

# Servicios disponibles:
# - API: http://localhost:8000
# - Query API: http://localhost:8001
# - Dashboard: http://localhost:8050
# - MLflow: http://localhost:5000
# - PostgreSQL: localhost:5435
```

### 3. Verificar Servicios

```bash
# API de Predicción
curl http://localhost:8000/health

# Query API
curl http://localhost:8001/health

# Dashboard
# Abrir navegador: http://localhost:8050

# MLflow UI
# Abrir navegador: http://localhost:5000
```

## Desarrollo

### Entrenar Modelos

```bash
# Entrenar FilmLens multi-label
cd src/filmlens
python train_pipeline.py

# Entrenar MovieLens multi-categoría
cd src/movielens
python train_pipeline.py
```

### Ejecutar APIs Localmente (sin Docker)

```bash
# API de Predicción
cd src/api
uvicorn main:app --reload --port 8000

# Query API
cd src/query_api
uvicorn main:app --reload --port 8001

# Dashboard
cd src/dashboard
python app.py
```

### Testing

```bash
# Ejecutar tests
pytest tests/

# Con coverage
pytest --cov=src tests/
```

### Modelos de Machine Learning

Los modelos entrenados se almacenan en la carpeta `models/production/` y son utilizados por los servicios:

```
models/
└── production/
    └── movielens/
        ├── multilevel_model_v2.pkl       
        ├── pipeline.pkl                   
        └── metadata.json                  
```

**Uso en Docker:**
- El `docker-compose.yml` monta la carpeta `models/production/movielens/` dentro del contenedor de la API
- El Dockerfile de la API copia los modelos al construir la imagen para deployment
- Los modelos son cargados por `src/api/models.py` al iniciar el servicio

**Generar nuevos modelos:**
```bash
# Entrenar y guardar modelo
cd src/movielens
python train_pipeline.py

# El modelo se guarda automáticamente en models/production/movielens/
# Reiniciar API para usar el nuevo modelo
docker-compose restart api
```

## Estructura de Base de Datos

La base de datos PostgreSQL contiene:

- **movies_catalog** - Catálogo de películas con metadata
- **predictions** - Predicciones históricas de sensibilidad
- **triggers** - Triggers detectados por película
- **user_preferences** - Preferencias de usuarios (futuro)

Las migraciones se gestionan con Flyway en `docker/migrations/`.

## Variables de Entorno Importantes

Se debe crear un archivo llamado `.env` se pude usar el `.env.example` como base.

```bash
# Base de Datos
DATABASE_URL=postgresql://user:pass@host:port/db
RDS_SECRET_MANAGER_ARN=arn:aws:secretsmanager:...

# AWS
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# APIs
API_BASE_URL=http://api:8000
QUERY_API_BASE_URL=http://query-api:8001
```

## Deployment

### Infraestructura como Código en Terraform

Este proyecto solo cubre la creacion de las imagenes en el registry de AWS.
También crea el bucket para versionar los datos con DVC. 

```bash
cd infrastructure/components/common
terraform init
terraform plan
terraform apply
```

## Monitoreo y Logs

- **MLflow UI**: http://localhost:5000 - Tracking de experimentos
- **Logs de contenedores**: `docker-compose logs -f <service>`
- **Métricas**: Almacenadas en MLflow backend (PostgreSQL)

## Versionado de Datos con DVC

Este proyecto usa DVC (Data Version Control) para versionar los datasets de entrenamiento y almacenarlos en S3.

### ¿Qué es DVC?

DVC es como Git pero para datos grandes. En lugar de guardar los archivos de datos en el repositorio, DVC guarda solo referencias pequeñas (archivos `.dvc`) mientras que los datos reales van a un almacenamiento remoto (en nuestro caso, S3).

### Configuración Inicial

El proyecto ya está configurado con DVC apuntando a un bucket S3. La configuración está en `.dvc/config`:

```ini
[core]
    remote = storage
['remote "storage"']
    url = s3://dvc-filmlens-data-repo
```

### Estructura de Datos Versionados

Los siguientes datasets están versionados con DVC:

```
data/
├── raw/
│   └── train/
│       ├── ml.dvc                    # MovieLens dataset
│       ├── movies_imdb_sample.dvc    # IMDB movies sample
│       └── consolidated.dvc          # Consolidated dataset
└── processed/
    ├── ml.dvc                        # Processed MovieLens
    └── movies_imdb_sample.dvc        # Processed IMDB sample
```

### Comandos Básicos de DVC

#### Descargar los datos (primera vez)

```bash
# Instalar DVC si no lo tienes
pip install dvc[s3]

# Configurar credenciales AWS (si no están configuradas)
aws configure

# Descargar todos los datos del proyecto
dvc pull
```

Esto descarga todos los archivos referenciados en los `.dvc` desde S3 a tu máquina local.

#### Actualizar datos locales

```bash
# Si alguien actualizó los datos en el repo
git pull
dvc pull
```

#### Añadir nuevos datos al versionado

```bash
# Agregar un nuevo dataset
dvc add data/raw/train/nuevo_dataset.csv

# Esto crea nuevo_dataset.csv.dvc
# Agregar el archivo .dvc al repo
git add data/raw/train/nuevo_dataset.csv.dvc
git add data/raw/train/.gitignore

# Subir el dataset real a S3
dvc push

# Commit y push del .dvc
git commit -m "agregar dataset"
git push
```

## Despliegue Continuo del Dashboard con CircleCI

El dashboard se despliega automáticamente en Railway usando CircleCI cuando se hace push a las ramas `main` o `first-pipeline-circleci`.

### Pipeline de Despliegue

El pipeline de CircleCI (`.circleci/config.yml`) ejecuta dos jobs:

1. **test_dashboard**: Valida el código del dashboard
   - Verifica formato con Black
   - Valida estructura de archivos y directorios
   - Instala dependencias desde `src/dashboard/requirements.txt`

2. **deploy_dashboard_to_railway**: Despliega a Railway
   - Se ejecuta solo si las pruebas pasan
   - Usa Railway CLI para deployment
   - Despliega el servicio configurado en Railway

### Configuración Requerida

En CircleCI, configurar la variable de entorno:
- `RAILWAY_TOKEN` - Token de autenticación de Railway

En Railway se required insertar las variables de entorno:

   - API_BASE_URL=http://IP o URL PUBLICA:8000
   - QUERY_API_BASE_URL=http://IP o URL PUBLICA:8001

## Notebooks y Scripts de Análisis

### Notebooks de Exploración (`notebooks/`)

Los notebooks de análisis exploratorio de datos (EDA) están en la carpeta `notebooks/`:

- **`01_eda_fuentes.ipynb`** - Análisis exploratorio de las fuentes de datos (MovieLens, IMDB)
- **`scripts/05_eda_sensibilidad.ipynb`** - EDA de sensibilidad y triggers
- **`scripts/06_data_processing_unification.py`** - Script de unificación de datasets
- **`scripts/film_sensitivity.py`** - Análisis de sensibilidad de películas
- **`scripts/modeltrain_filmasens00.py`** - Experimentos iniciales de entrenamiento

**Instalación de dependencias:**
```bash
pip install -r notebooks/requirements_eda.txt
```

### Scripts de Utilidad (`scripts/`)

Scripts para tareas específicas del proyecto:

- **`load_movies_catalog.py`** - Carga inicial del catálogo de películas a PostgreSQL
- **`load_movies_catalog.sh`** - Script bash para ejecutar la carga

**Uso:**
```bash
# Cargar catálogo de películas
python scripts/load_movies_catalog.py
# o usar el script bash
bash scripts/load_movies_catalog.sh
```

