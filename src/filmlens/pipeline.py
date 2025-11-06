from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import logging

from filmlens.config import config
from filmlens.processing import features

logger = logging.getLogger(__name__)


def create_pipeline() -> Pipeline:
    """
    Create sklearn pipeline from config.
    
    Returns:
        Pipeline ready for training
    """
    logger.info("Creating pipeline from config...")
    
    pipeline_steps = []
    
    # Step 1: Text cleaning
    pipeline_steps.append((
        'text_cleaner',
        features.TextCleaner()
    ))
    
    # Step 2: Genre encoding
    if config.feature_config.use_genre_features:
        pipeline_steps.append((
            'genre_encoder',
            features.GenreEncoder()
        ))
    
    # Step 3: Feature combination
    pipeline_steps.append((
        'feature_combiner',
        features.FeatureCombiner()
    ))
    
    # Step 4: Multi-output classifier
    rf_cfg = config.model.random_forest
    base_classifier = RandomForestClassifier(
        n_estimators=rf_cfg.n_estimators,
        max_depth=rf_cfg.max_depth,
        min_samples_split=rf_cfg.min_samples_split,
        min_samples_leaf=rf_cfg.min_samples_leaf,
        class_weight=rf_cfg.class_weight,
        random_state=rf_cfg.random_state,
        n_jobs=rf_cfg.n_jobs
    )
    
    multi_output_clf = MultiOutputClassifier(base_classifier)
    pipeline_steps.append(('classifier', multi_output_clf))
    
    pipeline = Pipeline(pipeline_steps)
    
    logger.info(f"Pipeline created with {len(pipeline_steps)} steps")
    return pipeline
