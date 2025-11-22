import requests
from typing import Dict, Optional, List
import logging

from config import (
    API_PREDICT_ENDPOINT,
    API_HEALTH_ENDPOINT,
    QUERY_API_AUTOCOMPLETE_ENDPOINT,
    QUERY_API_FILTERS_ENDPOINT,
    QUERY_API_MOVIE_DETAIL_ENDPOINT,
)

logger = logging.getLogger(__name__)


class APIClient:
    """Client for interacting with FilmLens APIs."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()

    def check_health(self) -> bool:
        """Check if the Prediction API is healthy and available."""
        try:
            response = self.session.get(API_HEALTH_ENDPOINT, timeout=5)
            return response.status_code == 200 and response.json().get("status") == "healthy"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def predict(self, movie_id: str, title: str, description: str, genre: str) -> Optional[Dict]:
        """
        Make a prediction request to the Prediction API.

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
                "verbose": True,
            }

            response = self.session.post(API_PREDICT_ENDPOINT, json=payload, timeout=self.timeout)

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

    def autocomplete_movies(self, query: str, limit: int = 10) -> Optional[List[Dict]]:
        """
        Get movie autocomplete suggestions from Query API.

        Args:
            query: Search term
            limit: Maximum number of suggestions

        Returns:
            List of movie suggestions or None if failed
        """
        try:
            params = {"q": query, "limit": limit}
            response = self.session.get(QUERY_API_AUTOCOMPLETE_ENDPOINT, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                return data.get("suggestions", [])
            else:
                logger.error(f"Autocomplete failed: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Autocomplete error: {e}")
            return None

    def get_filter_options(self) -> Optional[Dict]:
        """
        Get available filter options from Query API.

        Returns:
            Dict with genres, year_range, rating_range, runtime_range or None if failed
        """
        try:
            response = self.session.get(QUERY_API_FILTERS_ENDPOINT, timeout=10)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Get filters failed: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Get filters error: {e}")
            return None

    def get_movie_detail(self, movie_id: str) -> Optional[Dict]:
        """
        Get movie details from Query API.

        Args:
            movie_id: IMDB ID or TMDB ID

        Returns:
            Movie details dict or None if failed
        """
        try:
            url = f"{QUERY_API_MOVIE_DETAIL_ENDPOINT}/{movie_id}"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Get movie detail failed: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Get movie detail error: {e}")
            return None
