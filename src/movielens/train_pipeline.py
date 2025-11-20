"""
Script de entrenamiento del modelo de sensibilidad MovieLens.
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
import mlflow
import mlflow.sklearn
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))

from movielens.config import config
from movielens.processing import prepare_data, build_features, prepare_targets
from movielens.models import get_model


def evaluate_model(Y_test, Y_pred, Y_pred_proba, config):
    """
    Evalua el modelo por categoria y calcula metricas globales.
    Retorna diccionario con metricas para logging en MLflow.
    """
    target_columns = config['target_columns']
    target_classes = config['target_classes']
    n_labels = len(target_classes)
    
    print("\n")
    print("EVALUACION DEL MODELO MULTI-CATEGORIA")
    print("="*80)
    
    offset = 0
    mcauc_scores = []
    metrics = {}
    
    for i, cat in enumerate(target_columns):
        Y_test_cat = Y_test[:, offset:offset+n_labels]
        Y_pred_cat = Y_pred[:, offset:offset+n_labels]
        
        y_pred_proba_cat = np.column_stack([
            Y_pred_proba[offset + j][:, 1] for j in range(n_labels)
        ])
        
        print(f"\n{'-'*80}")
        print(f"CATEGORIA: {cat.upper()}")
        print(f"{'-'*80}")
        
        # Imprimir reporte
        print(classification_report(
            Y_test_cat, 
            Y_pred_cat, 
            target_names=target_classes, 
            zero_division=0
        ))
        
        # Obtener reporte como diccionario
        report = classification_report(
            Y_test_cat, 
            Y_pred_cat, 
            target_names=target_classes, 
            zero_division=0,
            output_dict=True
        )
        
        # Guardar metricas por categoria (usar samples avg en lugar de accuracy para multilabel)
        metrics[f"{cat}_macro_f1"] = report['macro avg']['f1-score']
        metrics[f"{cat}_weighted_f1"] = report['weighted avg']['f1-score']
        metrics[f"{cat}_samples_avg_f1"] = report['samples avg']['f1-score']
        
        valid_cols_cat = [j for j in range(n_labels) if len(np.unique(Y_test_cat[:, j])) > 1]
        if valid_cols_cat:
            try:
                mc_auc_cat = roc_auc_score(
                    Y_test_cat[:, valid_cols_cat], 
                    y_pred_proba_cat[:, valid_cols_cat], 
                    average='macro'
                )
                print(f"MCAUC (macro): {round(mc_auc_cat, 4)}")
                mcauc_scores.append(mc_auc_cat)
                metrics[f"{cat}_mcauc"] = mc_auc_cat
            except Exception:
                print("MCAUC: no calculable")
                metrics[f"{cat}_mcauc"] = 0.0
        else:
            print("MCAUC: no hay clases validas")
            metrics[f"{cat}_mcauc"] = 0.0
        
        offset += n_labels
    
    print(f"\n{'='*80}")
    print("RESUMEN GLOBAL")
    print(f"{'='*80}")
    
    if mcauc_scores:
        mcauc_promedio = np.mean(mcauc_scores)
        print(f"MCAUC promedio: {round(mcauc_promedio, 4)}")
        print(f"Categorias evaluadas: {len(mcauc_scores)} de {len(target_columns)}")
        
        print("\nMCAUC por categoria:")
        for cat, score in zip(target_columns[:len(mcauc_scores)], mcauc_scores):
            print(f"  - {cat}: {round(score, 4)}")
        
        metrics['avg_mcauc'] = mcauc_promedio
        metrics['categories_evaluated'] = len(mcauc_scores)
    else:
        print("No se pudo calcular MCAUC para ninguna categoria")
        metrics['avg_mcauc'] = 0.0
        metrics['categories_evaluated'] = 0
    
    return metrics


def save_model(model, transformers, config):
    """
    Guarda el modelo y los transformadores en disco.
    Retorna paths para registro en MLflow.
    """
    import joblib
    
    base_dir = Path(__file__).resolve().parents[2]
    models_dir = base_dir / "models" / "production" / "movielens"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline_name = config['pipeline_name']
    version = config['version']
    
    model_filename = f"{pipeline_name}_{version}.pkl"
    model_path = models_dir / model_filename
    joblib.dump(model, model_path)
    print(f"\nModelo guardado: {model_path}")
    
    tfidf_word_path = models_dir / f"tfidf_word_{version}.pkl"
    joblib.dump(transformers['tfidf_word'], tfidf_word_path)
    print(f"TF-IDF Word guardado: {tfidf_word_path}")
    
    tfidf_char_path = models_dir / f"tfidf_char_{version}.pkl"
    joblib.dump(transformers['tfidf_char'], tfidf_char_path)
    print(f"TF-IDF Char guardado: {tfidf_char_path}")
    
    sbert_path = None
    if transformers['sbert_model'] is not None:
        sbert_path = models_dir / f"sbert_model_{version}"
        transformers['sbert_model'].save(str(sbert_path))
        print(f"SBERT Model guardado: {sbert_path}")
    
    print(f"\nTodos los archivos guardados en: {models_dir}")
    
    return {
        'model_path': str(model_path),
        'tfidf_word_path': str(tfidf_word_path),
        'tfidf_char_path': str(tfidf_char_path),
        'sbert_path': str(sbert_path) if sbert_path else None
    }


def log_to_mlflow(model, transformers, metrics, paths, config, X_train, X_test):
    """
    Registra modelo, metricas y artifacts en MLflow.
    """
    mlflow_config = config['mlflow_config']
    
    # Configurar MLflow
    mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    mlflow.set_experiment(mlflow_config['experiment_name'])
    
    print("\n")
    print("REGISTRANDO EN MLFLOW")
    print("="*80)
    print(f"Tracking URI: {mlflow_config['tracking_uri']}")
    print(f"Experiment: {mlflow_config['experiment_name']}")
    
    version = config['version']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"movielens_{version}_{timestamp}"
    
    with mlflow.start_run(run_name=run_name):
        
        # Log parametros de configuracion
        mlflow.log_param("version", version)
        mlflow.log_param("algorithm", config['model_config']['algorithm'])
        mlflow.log_param("test_size", config['data_config']['test_size'])
        mlflow.log_param("random_state", config['data_config']['random_state'])
        
        # Log parametros de features
        mlflow.log_param("tfidf_word_max_features", config['feature_config']['tfidf_word']['max_features'])
        mlflow.log_param("tfidf_word_ngram", str(config['feature_config']['tfidf_word']['ngram_range']))
        mlflow.log_param("tfidf_char_max_features", config['feature_config']['tfidf_char']['max_features'])
        mlflow.log_param("sbert_model", config['feature_config']['sbert']['model_name'])
        mlflow.log_param("use_sbert", config['feature_config']['sbert']['use_sbert'])
        
        # Log parametros del algoritmo
        algorithm = config['model_config']['algorithm']
        algo_params = config['model_config'].get(algorithm, {})
        for param, value in algo_params.items():
            mlflow.log_param(f"{algorithm}_{param}", value)
        
        # Log dataset info
        mlflow.log_param("n_train", X_train.shape[0])
        mlflow.log_param("n_test", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_categories", len(config['target_columns']))
        mlflow.log_param("n_classes_per_category", len(config['target_classes']))
        
        # Log metricas
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log artifacts (archivos del modelo)
        print("\nRegistrando artifacts")
        mlflow.log_artifact(paths['model_path'], artifact_path="model")
        mlflow.log_artifact(paths['tfidf_word_path'], artifact_path="transformers")
        mlflow.log_artifact(paths['tfidf_char_path'], artifact_path="transformers")
        
        if paths['sbert_path']:
            mlflow.log_artifacts(paths['sbert_path'], artifact_path="transformers/sbert_model")
        
        # Registrar modelo en Model Registry
        print("\nRegistrando en Model Registry")
        
        # Crear wrapper personalizado para el pipeline completo
        import joblib
        
        # Empaquetar todo en un diccionario
        pipeline_artifacts = {
            'model': model,
            'tfidf_word': transformers['tfidf_word'],
            'tfidf_char': transformers['tfidf_char'],
            'sbert_model': transformers['sbert_model'],
            'config': config
        }
        
        # Log como sklearn model (MLflow detecta MultiOutputClassifier)
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline_artifacts,
            artifact_path="movielens_pipeline",
            registered_model_name=mlflow_config['registered_model_name']
        )
        
        # Tags para identificacion
        mlflow.set_tag("model_type", "multi-output-multiclass")
        mlflow.set_tag("framework", "sklearn")
        mlflow.set_tag("version", version)
        mlflow.set_tag("algorithm", algorithm)
        mlflow.set_tag("categories", ",".join(config['target_columns']))
        mlflow.set_tag("status", "trained")
        
        run_id = mlflow.active_run().info.run_id
        
        print(f"\nModelo registrado en MLflow")
        print(f"  Run ID: {run_id}")
        print(f"  Model URI: {model_info.model_uri}")
        print(f"  Registered Model: {mlflow_config['registered_model_name']}")
        print(f"  MCAUC Promedio: {metrics.get('avg_mcauc', 0):.4f}")


def main():
    """
    Pipeline principal de entrenamiento.
    """
    print("="*80)
    print("MOVIELENS - ENTRENAMIENTO DE MODELO DE SENSIBILIDAD")
    print("="*80)
    
    print("\n[1/6] Preparando datos")
    train_df, test_df = prepare_data(config)
    print(f"Train: {len(train_df)} registros")
    print(f"Test: {len(test_df)} registros")
    
    print("\n[2/6] Construyendo features")
    X_train, X_test, train_df, test_df, transformers = build_features(
        train_df, test_df, config
    )
    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape X_test: {X_test.shape}")
    
    print("\n[3/6] Preparando targets")
    Y_train, Y_test, mlb_dict = prepare_targets(train_df, test_df, config)
    print(f"Shape Y_train: {Y_train.shape}")
    print(f"Shape Y_test: {Y_test.shape}")
    
    print("\n[4/6] Entrenando modelo")
    algorithm = config['model_config']['algorithm']
    print(f"Algoritmo seleccionado: {algorithm}")
    
    model = get_model(algorithm, config)
    model.fit(X_train, Y_train)
    print("Entrenamiento completado")
    
    print("\n[5/6] Evaluando modelo")
    Y_pred = model.predict(X_test)
    
    Y_pred_proba = []
    n_labels = len(config['target_classes'])
    for i in range(Y_test.shape[1]):
        proba = model.model.estimators_[i].predict_proba(X_test)
        Y_pred_proba.append(proba)
    
    metrics = evaluate_model(Y_test, Y_pred, Y_pred_proba, config)
    
    paths = None
    if config['save_pipeline']:
        paths = save_model(model.model, transformers, config)
    
    print("\n[6/6] Registrando en MLflow")
    if paths:
        log_to_mlflow(model.model, transformers, metrics, paths, config, X_train, X_test)
    
    print("\n")
    print("ENTRENAMIENTO FINALIZADO")
    print("="*80)


if __name__ == "__main__":
    main()
