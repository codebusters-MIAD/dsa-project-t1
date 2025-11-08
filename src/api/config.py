from pathlib import Path
from pydantic_settings import BaseSettings
import glob


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
        """
        Obtener ruta completa del modelo.
        Busca el archivo mas reciente si no se especifica algoritmo.
        """
        if Path("/app").exists():
            model_dir = Path("/app/src/filmlens/trained_models")
        else:
            model_dir = self.project_root / "src/filmlens/trained_models"
        
        # Si el model_file no incluye algoritmo, buscar el mas reciente
        if "_random_forest" not in self.model_file and "_lightgbm" not in self.model_file and "_logistic_regression" not in self.model_file:
            # Buscar todos los modelos disponibles
            pattern = str(model_dir / "multilabel_classifier_v0.1.0_*.pkl")
            available_models = glob.glob(pattern)
            
            if available_models:
                # Usar el mas reciente
                model_path = Path(max(available_models, key=lambda p: Path(p).stat().st_mtime))
                return model_path
        
        return model_dir / self.model_file
    
    # CORS
    allowed_origins: list = ["*"]
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
