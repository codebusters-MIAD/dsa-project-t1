import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from filmlens.processing.validation import evaluate_multilabel_model
from filmlens.processing.features import create_keyword_labels

from filmlens.config import config
from filmlens.processing.data_manager import (
    load_dataset,
    pre_pipeline_preparation,
    save_pipeline,
    remove_old_pipelines
)
from filmlens.models import ModelRegistry
from filmlens import pipeline as ml_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_classifier():
    """Construir clasificador desde registry usando config."""
    algorithm_name = config.model.algorithm
    algorithm_params = config.model.get_algorithm_params()
    
    logger.info(f"Construyendo clasificador: {algorithm_name}")
    logger.info(f"Hiperparametros: {algorithm_params}")
    
    if not ModelRegistry.is_registered(algorithm_name):
        available = ModelRegistry.list_models()
        raise ValueError(
            f"Algoritmo '{algorithm_name}' no registrado. "
            f"Disponibles: {', '.join(available)}"
        )
    
    model_builder = ModelRegistry.get(algorithm_name, **algorithm_params)
    classifier = model_builder.get_model()
    
    return classifier, model_builder


def run_training() -> None:
    """Entrenar clasificador multi-label con algoritmo seleccionado."""
    
    logger.info("="*80)
    logger.info("FILMLENS MULTI-LABEL TRAINING PIPELINE")
    logger.info("="*80)
    
    mlflow.set_tracking_uri(config.app_config.mlflow_tracking_uri)
    mlflow.set_experiment(config.app_config.mlflow_experiment_name)
    
    classifier, model_builder = build_classifier()
    
    run_name = f"{config.model.algorithm}_training"
    
    with mlflow.start_run(run_name=run_name):
        
        mlflow.log_param("algorithm", config.model.algorithm)
        
        for key, value in model_builder.params.items():
            mlflow.log_param(f"model_{key}", value)
        
        for key, value in config.feature_config.tfidf.model_dump().items():
            mlflow.log_param(f"tfidf_{key}", value)
        
        logger.info("\n[1/6] Loading data...")
        df = load_dataset()
        df = pre_pipeline_preparation(df)
        
        mlflow.log_metric("total_samples", len(df))
        
        logger.info("\n[2/6] Preparing multi-label targets...")
        target_cols = config.data_config.target_columns
        
        
        df_with_labels = create_keyword_labels(df)
        
        y_multilabel = df_with_labels[target_cols].values
        X = df
        
        logger.info(f"Target shape: {y_multilabel.shape}")
        logger.info(f"Distribucion de triggers:")
        for i, col in enumerate(target_cols):
            pos_count = y_multilabel[:, i].sum()
            pos_pct = (pos_count / len(y_multilabel) * 100)
            logger.info(f"  {col}: {int(pos_count):,} ({pos_pct:.2f}%)")
        
        logger.info("\n[3/6] Splitting data...")
        stratify_col = config.data_config.stratify_by
        stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_multilabel,
            test_size=config.data_config.test_size,
            random_state=config.data_config.random_state,
            stratify=stratify
        )
        
        logger.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
        mlflow.log_metric("train_size", len(X_train))
        mlflow.log_metric("test_size", len(X_test))
        
        logger.info("\n[4/6] Construyendo pipeline ML...")
        pipeline = ml_pipeline.trigger_detection_pipeline(
            classifier=classifier,
            config=config
        )
        
        logger.info("\n[5/6] Entrenando modelo...")
        pipeline.fit(X_train, y_train)
        
        logger.info("\n[6/6] Evaluando modelo...")
        y_pred = pipeline.predict(X_test)
        
        
        metrics = evaluate_multilabel_model(
            y_test, 
            y_pred, 
            config.data_config.target_columns
        )
        
        logger.info(f"\nRendimiento del Modelo ({model_builder.name}):")
        logger.info(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
        logger.info(f"  Subset Accuracy: {metrics['subset_accuracy']:.4f}")
        logger.info(f"  F1-Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"  F1-Micro: {metrics['f1_micro']:.4f}")
        
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
        logger.info("\nGuardando modelo...")
        save_path = save_pipeline(
            pipeline_to_persist=pipeline,
            algorithm_name=config.model.algorithm
        )
        logger.info(f"Modelo guardado: {save_path}")
        
        # Mantener solo el modelo recien guardado
        remove_old_pipelines(files_to_keep=[save_path.name])
        
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            registered_model_name=config.app_config.package_name
        )
        
        run_id = mlflow.active_run().info.run_id
        logger.info(f"\n{'='*60}")
        logger.info(f"Entrenamiento completado exitosamente")
        logger.info(f"Algoritmo: {model_builder.name}")
        logger.info(f"MLflow Run ID: {run_id}")
        


if __name__ == "__main__":
    run_training()
