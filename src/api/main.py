import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .models import model_manager
from .database import db_manager
from .routers import health_router, predict_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Iniciando MovieLens Sensitivity API...")
    
    # Cargar modelo ML
    success = model_manager.load_model()
    
    if success:
        logger.info("Modelo cargado exitosamente")
    else:
        logger.error("Fallo al cargar modelo")
    
    # Base de datos deshabilitada por configuracion
    if settings.db_enabled:
        try:
            db_manager.initialize()
            logger.info("Conexion a base de datos inicializada")
        except Exception as e:
            logger.error(f"Fallo al inicializar base de datos: {e}")
    else:
        logger.info("Base de datos deshabilitada")
    
    yield
    
    # Shutdown
    logger.info("Cerrando MovieLens Sensitivity API...")
    if settings.db_enabled:
        db_manager.close()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description=settings.description,
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include routers
app.include_router(health_router)
app.include_router(predict_router)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
