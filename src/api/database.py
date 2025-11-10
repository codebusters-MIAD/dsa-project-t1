import logging
from contextlib import contextmanager
from typing import Optional
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

from .config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage PostgreSQL database connections and operations."""
    
    def __init__(self):
        self._pool: Optional[SimpleConnectionPool] = None
    
    def initialize(self):
        """Initialize database connection pool."""
        try:
            self._pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=settings.db_host,
                port=settings.db_port,
                database=settings.db_name,
                user=settings.db_user,
                password=settings.db_password
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    def close(self):
        """Close all database connections."""
        if self._pool:
            self._pool.closeall()
            logger.info("Database connection pool closed")
    
    @contextmanager
    def get_connection(self):
        """
        Get a database connection from the pool.
        
        Yields:
            psycopg2 connection object
        """
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        finally:
            if conn:
                self._pool.putconn(conn)
    
    def save_prediction(
        self,
        movie_id: str,
        title: str,
        description: Optional[str],
        predictions: dict,
        model_version: str,
        processing_time_ms: int
    ) -> bool:
        """
        Save prediction results to movie_triggers table.
        
        Args:
            movie_id: Unique movie identifier
            title: Movie title
            description: Movie description
            predictions: Dictionary with trigger predictions and confidences
            model_version: Model version used for prediction
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    query = """
                        INSERT INTO movie_triggers (
                            movie_id, title, description,
                            has_violence, has_sexual_content, has_substance_abuse,
                            has_suicide, has_strong_language,
                            violence_confidence, sexual_content_confidence,
                            substance_abuse_confidence, suicide_confidence,
                            strong_language_confidence,
                            model_version, processing_time_ms
                        ) VALUES (
                            %s, %s, %s,
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s,
                            %s, %s
                        )
                        ON CONFLICT (movie_id) 
                        DO UPDATE SET
                            title = EXCLUDED.title,
                            description = EXCLUDED.description,
                            detected_at = CURRENT_TIMESTAMP,
                            has_violence = EXCLUDED.has_violence,
                            has_sexual_content = EXCLUDED.has_sexual_content,
                            has_substance_abuse = EXCLUDED.has_substance_abuse,
                            has_suicide = EXCLUDED.has_suicide,
                            has_strong_language = EXCLUDED.has_strong_language,
                            violence_confidence = EXCLUDED.violence_confidence,
                            sexual_content_confidence = EXCLUDED.sexual_content_confidence,
                            substance_abuse_confidence = EXCLUDED.substance_abuse_confidence,
                            suicide_confidence = EXCLUDED.suicide_confidence,
                            strong_language_confidence = EXCLUDED.strong_language_confidence,
                            model_version = EXCLUDED.model_version,
                            processing_time_ms = EXCLUDED.processing_time_ms
                    """
                    
                    cursor.execute(query, (
                        movie_id, title, description,
                        predictions['has_violence'], predictions['has_sexual_content'],
                        predictions['has_substance_abuse'], predictions['has_suicide'],
                        predictions['has_strong_language'],
                        predictions['violence_confidence'], predictions['sexual_content_confidence'],
                        predictions['substance_abuse_confidence'], predictions['suicide_confidence'],
                        predictions['strong_language_confidence'],
                        model_version, processing_time_ms
                    ))
                    
                    conn.commit()
                    logger.info(f"Saved prediction for movie_id: {movie_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            return False
    
    def get_prediction(self, movie_id: str) -> Optional[dict]:
        """
        Retrieve prediction for a movie from database.
        
        Args:
            movie_id: Unique movie identifier
            
        Returns:
            Dictionary with prediction data or None if not found
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    query = """
                        SELECT * FROM movie_triggers 
                        WHERE movie_id = %s
                    """
                    cursor.execute(query, (movie_id,))
                    result = cursor.fetchone()
                    return dict(result) if result else None
                    
        except Exception as e:
            logger.error(f"Failed to get prediction: {e}")
            return None


# Global database manager instance
db_manager = DatabaseManager()
