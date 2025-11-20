import logging
import sys
from pathlib import Path

sys.path.insert(0, '/app/src')

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

from .config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Gestiona el pipeline completo de MovieLens: modelo + transformadores"""
    
    def __init__(self):
        self._model = None
        self._tfidf_word = None
        self._tfidf_char = None
        self._sbert_model = None
        self._model_metadata = {}
        
        # Configuracion de categorias y clases
        self.target_columns = [
            'violencia_nivel',
            'sexualidad_nivel', 
            'drogas_nivel',
            'lenguaje_fuerte_nivel',
            'suicidio_nivel'
        ]
        self.target_classes = ['sin_contenido', 'moderado', 'alto']
    
    def load_model(self) -> bool:
        """
        Carga el modelo y todos los transformadores necesarios.
        
        Returns:
            True si exitoso, False en caso contrario
        """
        try:
            # Verificar rutas
            model_path = settings.model_path
            tfidf_word_path = settings.tfidf_word_path
            tfidf_char_path = settings.tfidf_char_path
            sbert_path = settings.sbert_path
            
            if not model_path.exists():
                logger.error(f"Modelo no encontrado: {model_path}")
                return False
            
            if not tfidf_word_path.exists():
                logger.error(f"TF-IDF word no encontrado: {tfidf_word_path}")
                return False
                
            if not tfidf_char_path.exists():
                logger.error(f"TF-IDF char no encontrado: {tfidf_char_path}")
                return False
            
            if not sbert_path.exists():
                logger.error(f"SBERT no encontrado: {sbert_path}")
                return False
            
            # Cargar modelo clasificador
            logger.info(f"Cargando modelo desde: {model_path}")
            self._model = joblib.load(model_path)
            
            # Cargar transformadores TF-IDF
            logger.info(f"Cargando TF-IDF word desde: {tfidf_word_path}")
            self._tfidf_word = joblib.load(tfidf_word_path)
            
            logger.info(f"Cargando TF-IDF char desde: {tfidf_char_path}")
            self._tfidf_char = joblib.load(tfidf_char_path)
            
            # Cargar SBERT
            logger.info(f"Cargando SBERT desde: {sbert_path}")
            self._sbert_model = SentenceTransformer(str(sbert_path))
            
            self._model_metadata = {
                "version": settings.model_version,
                "model_path": str(model_path),
                "tfidf_word_path": str(tfidf_word_path),
                "tfidf_char_path": str(tfidf_char_path),
                "sbert_path": str(sbert_path),
                "loaded": True
            }
            
            logger.info("Pipeline completo cargado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            return False
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocesa texto (limpieza basica).
        El preprocesamiento completo lo hacen los vectorizadores TF-IDF.
        """
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_features(self, text: str):
        """
        Extrae features usando TF-IDF word, char y SBERT.
        
        Returns:
            Sparse matrix con todas las features concatenadas
        """
        if not self.is_loaded:
            raise RuntimeError("Modelo no cargado")
        
        # Preprocesar
        text_clean = self._preprocess_text(text)
        
        # TF-IDF word-level
        X_tfidf_word = self._tfidf_word.transform([text_clean])
        
        # TF-IDF char-level
        X_tfidf_char = self._tfidf_char.transform([text_clean])
        
        # SBERT embeddings
        sbert_embedding = self._sbert_model.encode([text_clean])
        X_sbert = csr_matrix(sbert_embedding)
        
        # Concatenar features
        X_features = hstack([X_tfidf_word, X_tfidf_char, X_sbert])
        
        return X_features
    
    @property
    def is_loaded(self) -> bool:
        """Verifica si el pipeline esta cargado"""
        return (self._model is not None and 
                self._tfidf_word is not None and
                self._tfidf_char is not None and
                self._sbert_model is not None)
    
    @property
    def metadata(self) -> dict:
        """Retorna metadata del modelo"""
        return self._model_metadata
    
    def predict(self, description: str) -> dict:
        """
        Predice niveles de sensibilidad para una descripcion.
        
        Args:
            description: Texto de descripcion de la pelicula
            
        Returns:
            Dict con predicciones por categoria
        """
        if not self.is_loaded:
            raise RuntimeError("Modelo no cargado")
        
        # Extraer features
        X = self._extract_features(description)
        
        # Predecir
        y_pred = self._model.predict(X)[0]
        
        # Obtener probabilidades
        y_proba = []
        for estimator in self._model.estimators_:
            proba = estimator.predict_proba(X)[0]
            y_proba.append(proba)
        
        # Formatear resultados por categoria
        predictions = {}
        n_classes = len(self.target_classes)
        
        for i, category in enumerate(self.target_columns):
            offset = i * n_classes
            category_pred = y_pred[offset:offset + n_classes]
            category_proba = y_proba[offset:offset + n_classes]
            
            # Encontrar clase predicha
            predicted_idx = np.argmax(category_pred)
            predicted_class = self.target_classes[predicted_idx]
            
            # Extraer probabilidades por clase
            class_probabilities = {}
            for j, class_name in enumerate(self.target_classes):
                prob = float(category_proba[j][1]) if len(category_proba[j]) > 1 else 0.0
                class_probabilities[class_name] = prob
            
            predictions[category] = {
                'nivel': predicted_class,
                'probabilidad': class_probabilities[predicted_class],
                'probabilidades_todas': class_probabilities
            }
        
        return predictions


model_manager = ModelManager()
