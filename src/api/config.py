from pathlib import Path
from pydantic_settings import BaseSettings
import glob


class Settings(BaseSettings):
    # API metadata
    app_name: str = "FilmLens Sensitivity Classification API"
    version: str = "1.0.0"
    description: str = "Multi-output multiclass API for content sensitivity classification in movies"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Database
    db_enabled: bool = True
    db_host: str = "filmlens-db"
    db_port: int = 5432
    db_name: str = "filmlens"
    db_user: str = "filmlens_user"
    db_password: str = "filmlens_dev_2025"
    
    # Model version
    model_version: str = "v1.0.0"
    
    # Paths
    project_root: Path = Path(__file__).parent.parent.parent
    
    @property
    def models_dir(self) -> Path:
        """Directorio base de modelos"""
        if Path("/app").exists():
            return Path("/app/models/production/movielens")
        return self.project_root / "models" / "production" / "movielens"
    
    @property
    def model_path(self) -> Path:
        return self.models_dir / f"movielens_sensitivity_classifier_{self.model_version}.pkl"
    
    @property
    def tfidf_word_path(self) -> Path:
        return self.models_dir / f"tfidf_word_{self.model_version}.pkl"
    
    @property
    def tfidf_char_path(self) -> Path:
        return self.models_dir / f"tfidf_char_{self.model_version}.pkl"
    
    @property
    def sbert_path(self) -> Path:
        return self.models_dir / f"sbert_model_{self.model_version}"
    
    # CORS
    allowed_origins: list = ["*"]
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
