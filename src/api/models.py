import logging
from pathlib import Path
from typing import Optional
import sys

# Add /app/src to sys.path to allow joblib to find 'filmlens' during unpickling
sys.path.insert(0, '/app/src')

import joblib

from .config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manage ML model lifecycle."""
    
    def __init__(self):
        self._model = None
        self._model_metadata = {}
    
    def load_model(self) -> bool:
        """
        Load the trained model from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = settings.full_model_path
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            logger.info(f"Loading model from: {model_path}")
            self._model = joblib.load(model_path)
            
            self._model_metadata = {
                "file_name": settings.model_file,
                "path": str(model_path),
                "loaded": True
            }
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    @property
    def model(self):
        """Get the loaded model."""
        return self._model
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    @property
    def metadata(self) -> dict:
        """Get model metadata."""
        return self._model_metadata
    
    def predict(self, input_data):
        """Make predictions using the loaded model."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        return self._model.predict(input_data)
    
    def predict_proba(self, input_data):
        """Get prediction probabilities."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        return self._model.predict_proba(input_data)


model_manager = ModelManager()
