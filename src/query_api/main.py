"""
FilmLens Query API - Database Query Service
FastAPI service for querying movie_triggers table
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import triggers
from .database import engine, Base

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Starting FilmLens Query API...")
    
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified")
    
    yield
    
    logger.info("Shutting down FilmLens Query API...")


app = FastAPI(
    title="FilmLens Query API",
    description="API for querying movie triggers and predictions",
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
