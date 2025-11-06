import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API metadata
    app_name: str = "FilmLens Trigger Detection API"
    version: str = "0.1.0"
    description: str = "Multi-label classification API for detecting sensitive content triggers in movies"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Model
    model_file: str = "multilabel_classifier_v0.1.0.pkl"
    
    # Paths
    project_root: Path = Path(__file__).parent.parent.parent
    
    @property
    def full_model_path(self) -> Path:
        # In Docker, model is in /app/src/filmlens/trained_models
        # Locally, model is in src/filmlens/trained_models
        model_path = Path("/app/src/filmlens/trained_models") if Path("/app").exists() else self.project_root / "src/filmlens/trained_models"
        return model_path / self.model_file
    
    # CORS
    allowed_origins: list = ["*"]
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
