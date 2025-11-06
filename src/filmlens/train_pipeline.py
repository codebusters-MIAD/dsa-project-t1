import logging
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    hamming_loss, 
    accuracy_score,
    f1_score,
    classification_report
)
import numpy as np

from filmlens.config import config
from filmlens.processing.data_manager import (
    load_dataset,
    pre_pipeline_preparation,
    save_pipeline,
    remove_old_pipelines
)
from filmlens.pipeline import create_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_training() -> None:
    """Execute training pipeline."""
    
    logger.info("="*80)
    logger.info("FILMLENS MULTI-LABEL TRAINING PIPELINE")
    logger.info("="*80)
    
    # Setup MLflow
    mlflow.set_tracking_uri(config.app_config.mlflow_tracking_uri)
    mlflow.set_experiment(config.app_config.mlflow_experiment_name)
    
    with mlflow.start_run(run_name="multilabel_training"):
        
        # Log config parameters
        mlflow.log_params({
            "algorithm": config.model.algorithm,
            "n_estimators": config.model.random_forest.n_estimators,
            "max_depth": config.model.random_forest.max_depth,
            "test_size": config.data_config.test_size,
            "tfidf_max_features": config.feature_config.tfidf.max_features
        })
        
        # 1. Load data
        logger.info("\n[1/6] Loading data...")
        df = load_dataset()
        df = pre_pipeline_preparation(df)
        
        mlflow.log_metric("total_samples", len(df))
        
        # 2. Prepare multi-label targets
        logger.info("\n[2/6] Preparing multi-label targets...")
        target_cols = config.data_config.target_columns
        
        # Create targets from keywords
        from filmlens.processing.features import create_keyword_labels
        df_with_labels = create_keyword_labels(df)
        
        y_multilabel = df_with_labels[target_cols].values

        X = df
        
        logger.info(f"Target shape: {y_multilabel.shape}")
        logger.info(f"Trigger distribution:")
        for i, col in enumerate(target_cols):
            pos_count = y_multilabel[:, i].sum()
            pos_pct = (pos_count / len(y_multilabel) * 100)
            logger.info(f"  {col}: {int(pos_count):,} ({pos_pct:.2f}%)")
        
        # 3. Train/test split
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
        
        # 4. Create and train pipeline
        logger.info("\n[4/6] Creating pipeline...")
        pipeline = create_pipeline()
        
        logger.info("\n[5/6] Training model...")
        pipeline.fit(X_train, y_train)
        
        # 5. Evaluate
        logger.info("\n[6/6] Evaluating model...")
        y_pred = pipeline.predict(X_test)
        
        # Global metrics
        hamming = hamming_loss(y_test, y_pred)
        subset_acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
        
        metrics = {
            'hamming_loss': hamming,
            'subset_accuracy': subset_acc,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro
        }
        
        mlflow.log_metrics(metrics)
        
        logger.info(f"\nGlobal Metrics:")
        logger.info(f"  Hamming Loss: {hamming:.4f}")
        logger.info(f"  Subset Accuracy: {subset_acc:.4f}")
        logger.info(f"  F1-Macro: {f1_macro:.4f}")
        logger.info(f"  F1-Micro: {f1_micro:.4f}")
        
        # Per-trigger metrics
        logger.info(f"\nPer-Trigger Metrics:")
        for i, trigger_name in enumerate(target_cols):
            y_true_trigger = y_test[:, i]
            y_pred_trigger = y_pred[:, i]
            
            f1 = f1_score(y_true_trigger, y_pred_trigger, zero_division=0)
            logger.info(f"  {trigger_name}: F1={f1:.4f}")
            
            mlflow.log_metric(f"f1_{trigger_name}", f1)
        
        # 6. Save pipeline
        logger.info("\nSaving pipeline...")
        save_pipeline(pipeline_to_persist=pipeline)
        remove_old_pipelines()
        
        # Log model to MLflow
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            registered_model_name=config.app_config.package_name
        )
        
        logger.info("\nTraining completed successfully!")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    run_training()
