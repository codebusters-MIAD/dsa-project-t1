# dsa-project-t1
FilmLens

1) Problema que abordaremos y contexto

Hoy es difícil saber qué temas sensibles aparecen en una película antes de verla. La información está dispersa (TMDb, OMDb, reseñas, sinopsis) y no existe una guía clara, en español y sin spoilers, que ayude a evitar contenido que pueda generar malestar emocional (p. ej., violencia sexual, suicidio/autolesión, abuso infantil, drogas, violencia gráfica, discriminación/odio, lenguaje fuerte, terror psicológico, maltrato animal).
CineConsciente propone una herramienta que analiza sinopsis y metadatos, detecta disparadores de contenido (multietiqueta), asigna niveles (leve/medio/alto) y calcula un Content Safety Score (0–100). El resultado se muestra en un dashboard y se expone vía API para integraciones.

2) Pregunta de negocio y alcance del proyecto

Pregunta: ¿Podemos ayudar a las personas a elegir qué ver con mayor tranquilidad ofreciendo alertas explicables (sin spoilers) sobre temas sensibles en películas, y filtros personalizables por preferencias?

Alcance (MVP – 4 semanas):

Fuentes: TMDb (API) y OMDb (API) → títulos, sinopsis, géneros, país, año, idioma, rating.

NLP: vectorización (TF-IDF) y reglas/umbrales por dimensión; clasificación multietiqueta + niveles (L/M/H); frases de soporte.

Datos: almacenados en Parquet y versionados con DVC (remoto S3).

Modelos: entrenados en Python/Scikit-learn, exportados a ONNX para inferencia rápida/portátil.

MLOps: MLflow (tracking + Model Registry), GitHub Actions (CI), FastAPI (serving), Streamlit/Dash (dashboard básico).

No se procesan datos personales.

3) Estructura inicial de carpetas (pensada para DataOps/MLOps)

Recomendación: mantener notebooks en una carpeta aparte para exploración (EDA, prototipos). La lógica estable pasa luego a src/. Puedes usar Jupytext para parear .ipynb ↔ .py y evitar diffs ruidosos.


src/ es para el código fuente.
data/ es para todas las versiones del conjunto de datos.
data/raw/ es para datos obtenidos de una fuente externa.
data/prepared/ es para datos modificados internamente.
model/ es para modelos de aprendizaje automático.
data/metrics/ sirve para realizar un seguimiento de las métricas de rendimiento de sus modelos.

La carpeta src/ contiene tres archivos Python:

prepare.py contiene código para preparar datos para el entrenamiento.
train.py contiene código para entrenar un modelo de aprendizaje automático.
evaluate.py contiene código para evaluar los resultados de un modelo de aprendizaje automático.


Se investigo una posible estructura del proyecto https://es.python-3.com/?p=283 y se complemento navegando por internet
```bash
data-version-control/
|
├── data/
│   ├── prepared/
│   └── raw/
|
├── metrics/
├── model/
└── src/
    ├── evaluate.py
    ├── prepare.py
    └── train.py
```
