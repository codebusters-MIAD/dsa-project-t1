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
        Save prediction results to movie_triggers table (multilevel schema V7).
        
        Args:
            movie_id: Unique movie identifier
            title: Movie title
            description: Movie description
            predictions: Dictionary with multilevel predictions per category
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
                            violencia_nivel, violencia_probabilidad,
                            violencia_prob_sin_contenido, violencia_prob_moderado, violencia_prob_alto,
                            sexualidad_nivel, sexualidad_probabilidad,
                            sexualidad_prob_sin_contenido, sexualidad_prob_moderado, sexualidad_prob_alto,
                            drogas_nivel, drogas_probabilidad,
                            drogas_prob_sin_contenido, drogas_prob_moderado, drogas_prob_alto,
                            lenguaje_fuerte_nivel, lenguaje_fuerte_probabilidad,
                            lenguaje_fuerte_prob_sin_contenido, lenguaje_fuerte_prob_moderado, lenguaje_fuerte_prob_alto,
                            suicidio_nivel, suicidio_probabilidad,
                            suicidio_prob_sin_contenido, suicidio_prob_moderado, suicidio_prob_alto,
                            model_version, processing_time_ms
                        ) VALUES (
                            %s, %s, %s,
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s,
                            %s, %s
                        )
                        ON CONFLICT (movie_id) 
                        DO UPDATE SET
                            title = EXCLUDED.title,
                            description = EXCLUDED.description,
                            detected_at = CURRENT_TIMESTAMP,
                            violencia_nivel = EXCLUDED.violencia_nivel,
                            violencia_probabilidad = EXCLUDED.violencia_probabilidad,
                            violencia_prob_sin_contenido = EXCLUDED.violencia_prob_sin_contenido,
                            violencia_prob_moderado = EXCLUDED.violencia_prob_moderado,
                            violencia_prob_alto = EXCLUDED.violencia_prob_alto,
                            sexualidad_nivel = EXCLUDED.sexualidad_nivel,
                            sexualidad_probabilidad = EXCLUDED.sexualidad_probabilidad,
                            sexualidad_prob_sin_contenido = EXCLUDED.sexualidad_prob_sin_contenido,
                            sexualidad_prob_moderado = EXCLUDED.sexualidad_prob_moderado,
                            sexualidad_prob_alto = EXCLUDED.sexualidad_prob_alto,
                            drogas_nivel = EXCLUDED.drogas_nivel,
                            drogas_probabilidad = EXCLUDED.drogas_probabilidad,
                            drogas_prob_sin_contenido = EXCLUDED.drogas_prob_sin_contenido,
                            drogas_prob_moderado = EXCLUDED.drogas_prob_moderado,
                            drogas_prob_alto = EXCLUDED.drogas_prob_alto,
                            lenguaje_fuerte_nivel = EXCLUDED.lenguaje_fuerte_nivel,
                            lenguaje_fuerte_probabilidad = EXCLUDED.lenguaje_fuerte_probabilidad,
                            lenguaje_fuerte_prob_sin_contenido = EXCLUDED.lenguaje_fuerte_prob_sin_contenido,
                            lenguaje_fuerte_prob_moderado = EXCLUDED.lenguaje_fuerte_prob_moderado,
                            lenguaje_fuerte_prob_alto = EXCLUDED.lenguaje_fuerte_prob_alto,
                            suicidio_nivel = EXCLUDED.suicidio_nivel,
                            suicidio_probabilidad = EXCLUDED.suicidio_probabilidad,
                            suicidio_prob_sin_contenido = EXCLUDED.suicidio_prob_sin_contenido,
                            suicidio_prob_moderado = EXCLUDED.suicidio_prob_moderado,
                            suicidio_prob_alto = EXCLUDED.suicidio_prob_alto,
                            model_version = EXCLUDED.model_version,
                            processing_time_ms = EXCLUDED.processing_time_ms
                    """
                    
                    # Extraer valores del dict de predicciones
                    violencia = predictions['violencia_nivel']
                    sexualidad = predictions['sexualidad_nivel']
                    drogas = predictions['drogas_nivel']
                    lenguaje = predictions['lenguaje_fuerte_nivel']
                    suicidio = predictions['suicidio_nivel']
                    
                    cursor.execute(query, (
                        movie_id, title, description,
                        violencia['nivel'], violencia['probabilidad'],
                        violencia['probabilidades_todas']['sin_contenido'],
                        violencia['probabilidades_todas']['moderado'],
                        violencia['probabilidades_todas']['alto'],
                        sexualidad['nivel'], sexualidad['probabilidad'],
                        sexualidad['probabilidades_todas']['sin_contenido'],
                        sexualidad['probabilidades_todas']['moderado'],
                        sexualidad['probabilidades_todas']['alto'],
                        drogas['nivel'], drogas['probabilidad'],
                        drogas['probabilidades_todas']['sin_contenido'],
                        drogas['probabilidades_todas']['moderado'],
                        drogas['probabilidades_todas']['alto'],
                        lenguaje['nivel'], lenguaje['probabilidad'],
                        lenguaje['probabilidades_todas']['sin_contenido'],
                        lenguaje['probabilidades_todas']['moderado'],
                        lenguaje['probabilidades_todas']['alto'],
                        suicidio['nivel'], suicidio['probabilidad'],
                        suicidio['probabilidades_todas']['sin_contenido'],
                        suicidio['probabilidades_todas']['moderado'],
                        suicidio['probabilidades_todas']['alto'],
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
