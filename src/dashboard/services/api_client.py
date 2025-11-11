import requests
from typing import Dict, Optional
import logging

from config import API_PREDICT_ENDPOINT, API_HEALTH_ENDPOINT

logger = logging.getLogger(__name__)


class APIClient:
    """Client for interacting with FilmLens API."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
    
    def check_health(self) -> bool:
        """Check if the API is healthy and available."""
        try:
            response = self.session.get(API_HEALTH_ENDPOINT, timeout=5)
            return response.status_code == 200 and response.json().get('status') == 'healthy'
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def predict(
        self, 
        movie_id: str, 
        title: str, 
        description: str, 
        genre: str
    ) -> Optional[Dict]:
        """
        Make a prediction request to the API.
        
        Args:
            movie_id: Unique movie identifier
            title: Movie title
            description: Movie plot description
            genre: Primary genre
            
        Returns:
            Prediction response dict or None if failed
        """
        try:
            payload = {
                "movie_id": movie_id,
                "title": title,
                "description": description,
                "genre": genre,
                "verbose": True
            }
            
            response = self.session.post(
                API_PREDICT_ENDPOINT,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Prediction failed: HTTP {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("Request timeout")
            return None
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
