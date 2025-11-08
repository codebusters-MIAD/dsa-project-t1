"""Pipeline de deteccion de triggers con clasificadores configurables."""

from sklearn.pipeline import Pipeline
import logging

from filmlens.config import config
from filmlens.processing import features

logger = logging.getLogger(__name__)


def trigger_detection_pipeline(classifier, config) -> Pipeline:
    """
    Crear pipeline de sklearn con clasificador parametrizable.
    
    Args:
        classifier: Instancia del clasificador (e.g., MultiOutputClassifier)
        config: Objeto de configuracion
        
    Returns:
        Pipeline listo para entrenamiento
    """
    logger.info("Creando pipeline desde config...")
    
    pipeline_steps = []
    
    pipeline_steps.append((
        'text_cleaner',
        features.TextCleaner()
    ))
    
    if config.feature_config.use_genre_features:
        pipeline_steps.append((
            'genre_encoder',
            features.GenreEncoder()
        ))
    
    pipeline_steps.append((
        'feature_combiner',
        features.FeatureCombiner()
    ))
    
    pipeline_steps.append(('classifier', classifier))
    
    pipeline = Pipeline(pipeline_steps)
    
    logger.info(f"Pipeline creado con {len(pipeline_steps)} pasos")
    return pipeline
