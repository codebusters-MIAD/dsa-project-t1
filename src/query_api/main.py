import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import triggers, movies
from .database import engine, Base

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FilmLens Query API...")
    
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified")
    
    yield
    
    logger.info("Shutting down FilmLens Query API...")


app = FastAPI(
    title="FilmLens Query API",
    description="API for querying movie triggers and predictions from FilmLens database",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(movies.router)
app.include_router(triggers.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "FilmLens Query API",
        "version": "0.1.0",
        "description": "Query service for movie triggers database"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "query-api",
        "version": "0.1.0"
    }
