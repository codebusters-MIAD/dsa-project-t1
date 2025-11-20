"""
Script de inferencia para el modelo MovieLens.
"""

import sys
import numpy as np
import joblib
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from movielens.config import config
from movielens.processing.features import clean_tokenize, combine_features


def load_model_and_transformers(version):
    """
    Carga el modelo y transformadores entrenados.
    """
    base_dir = Path(__file__).resolve().parents[2]
    models_dir = base_dir / "models" / "production" / "movielens"
    
    pipeline_name = config['pipeline_name']
    
    model_path = models_dir / f"{pipeline_name}_{version}.pkl"
    tfidf_word_path = models_dir / f"tfidf_word_{version}.pkl"
    tfidf_char_path = models_dir / f"tfidf_char_{version}.pkl"
    
    print(f"Cargando modelo desde: {model_path}")
    model = joblib.load(model_path)
    
    print(f"Cargando transformadores TF-IDF...")
    tfidf_word = joblib.load(tfidf_word_path)
    tfidf_char = joblib.load(tfidf_char_path)
    
    sbert_model = None
    if config['features_config']['use_sbert']:
        from sentence_transformers import SentenceTransformer
        sbert_path = models_dir / f"sbert_model_{version}"
        print(f"Cargando SBERT desde: {sbert_path}")
        sbert_model = SentenceTransformer(str(sbert_path))
    
    transformers = {
        'tfidf_word': tfidf_word,
        'tfidf_char': tfidf_char,
        'sbert_model': sbert_model
    }
    
    return model, transformers


def predict_single_description(description, model, transformers):
    """
    Realiza prediccion para una sola descripcion de pelicula.
    """
    text_clean = clean_tokenize(description, config)
    
    X_tfidf_word = transformers['tfidf_word'].transform([text_clean])
    X_tfidf_char = transformers['tfidf_char'].transform([text_clean])
    
    if transformers['sbert_model'] is not None:
        sbert_emb = transformers['sbert_model'].encode([text_clean])
        X_features = combine_features(X_tfidf_word, X_tfidf_char, sbert_emb)
    else:
        X_features = combine_features(X_tfidf_word, X_tfidf_char, None)
    
    y_pred = model.predict(X_features)[0]
    
    y_proba_list = []
    for estimator in model.estimators_:
        proba = estimator.predict_proba(X_features)[0]
        y_proba_list.append(proba[1] if len(proba) > 1 else 0.0)
    
    return y_pred, y_proba_list


def format_predictions(y_pred, y_proba, config):
    """
    Formatea las predicciones en un diccionario legible.
    """
    target_columns = config['target_columns']
    target_classes = config['target_classes']
    n_labels = len(target_classes)
    
    predictions = {}
    offset = 0
    
    for cat in target_columns:
        cat_pred = y_pred[offset:offset+n_labels]
        cat_proba = y_proba[offset:offset+n_labels]
        
        nivel_detectado = None
        for idx, label in enumerate(cat_pred):
            if label == 1:
                nivel_detectado = target_classes[idx]
                probabilidad = round(cat_proba[idx], 4)
                break
        
        if nivel_detectado is None:
            nivel_detectado = "sin_contenido"
            probabilidad = round(cat_proba[0], 4) if cat_proba[0] > 0 else 0.5
        
        predictions[cat] = {
            'nivel': nivel_detectado,
            'probabilidad': probabilidad,
            'probabilidades_todas': {
                target_classes[i]: round(cat_proba[i], 4)
                for i in range(n_labels)
            }
        }
        
        offset += n_labels
    
    return predictions


def main():
    """
    Funcion principal para prueba interactiva.
    """
    print("="*80)
    print("MOVIELENS - PREDICCION DE SENSIBILIDAD")
    print("="*80)
    
    version = config.get('version', 'v1.0.0')
    
    print(f"\nCargando modelo version: {version}")
    model, transformers = load_model_and_transformers(version)
    print("Modelo cargado correctamente\n")
    
    print("Ingrese la descripcion de la pelicula (o 'salir' para terminar):")
    print("-"*80)
    
    while True:
        description = input("\nDescripcion: ").strip()
        
        if description.lower() == 'salir':
            print("\nFinalizando...")
            break
        
        if not description:
            print("Por favor ingrese una descripcion valida")
            continue
        
        y_pred, y_proba = predict_single_description(description, model, transformers)
        predictions = format_predictions(y_pred, y_proba, config)
        
        print("\n" + "="*80)
        print("RESULTADOS DE LA PREDICCION")
        print("="*80)
        
        for categoria, resultado in predictions.items():
            print(f"\n{categoria.upper()}")
            print(f"  Nivel: {resultado['nivel']}")
            print(f"  Probabilidad: {resultado['probabilidad']}")
            print(f"  Todas las probabilidades:")
            for nivel, prob in resultado['probabilidades_todas'].items():
                print(f"    - {nivel}: {prob}")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    main()
