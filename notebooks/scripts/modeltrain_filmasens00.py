import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import os
import numpy as np
import csv
import ast
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from collections import Counter
from itertools import chain
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix




ruta = "../../data/processed/ml/dataset_sensibilidad_imdb_final_complete.csv"

def leer_csv_robusto(path: str) -> pd.DataFrame:
    
    candidatos = [';', ',', '\t', '|']
    for sep in candidatos:
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                engine='python',           
                encoding='utf-8-sig',      
                quotechar='"',             
                doublequote=True,
            )
            print(f" Cargado con separador explícito: '{sep}' → columnas: {len(df.columns)}, filas: {len(df)}")
            return df
        except Exception as e:
            
            pass

    
    try:
        with open(path, "r", encoding="utf-8-sig", newline='') as f:
            muestra = f.read(4096)
            dialecto = csv.Sniffer().sniff(muestra, delimiters=[',',';','\t','|'])
            sep_detectado = dialecto.delimiter
        df = pd.read_csv(
            path,
            sep=sep_detectado,
            engine='python',
            encoding='utf-8-sig',
            quotechar='"',
            doublequote=True,
        )
        print(f" Cargado con delimitador detectado automáticamente: '{sep_detectado}' → columnas: {len(df.columns)}, filas: {len(df)}")
        return df
    except Exception as e:
        print(" Falla en detección automática, se intentará omitir líneas problemáticas...")

    
    
    try:
        df = pd.read_csv(
            path,
            sep=None,                   
            engine='python',
            encoding='utf-8-sig',
            quotechar='"',
            doublequote=True,
            on_bad_lines='skip'         
        )
        print(f" Cargado omitiendo líneas problemáticas. Filas: {len(df)}, Columnas: {len(df.columns)}")
        return df
    except Exception as e:
        raise RuntimeError(f" No fue posible cargar el CSV: {e}")

# Ejecutar carga
dataTraining = df = leer_csv_robusto(ruta)

dataTraining.info()



nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return words

dataTraining['clean_tokens'] = dataTraining['description'].apply(clean_tokenize)
dataTraining['clean_text'] = dataTraining['clean_tokens'].apply(lambda x: ' '.join(x))
dataTraining['plot_length'] = dataTraining['clean_tokens'].apply(len)

dataTraining = dataTraining[dataTraining['plot_length'] >= 5].reset_index(drop=True)

NIVELES_VALIDOS = {"sin_contenido", "moderado", "alto"}

niv = dataTraining['violencia_nivel'].astype(str).str.strip().str.lower()

niv = niv.replace({"nan": "", "none": "", "": ""})

dataTraining['violencia_nivel_list'] = niv.apply(
    lambda s: [s] if s in NIVELES_VALIDOS else []
)



df_major = dataTraining[dataTraining['violencia_nivel'] == 'sin_contenido']
df_med = dataTraining[dataTraining['violencia_nivel'] == 'moderado']
df_high = dataTraining[dataTraining['violencia_nivel'] == 'alto']

df_med_upsampled = resample(df_med, replace=True, n_samples=len(df_major)//2, random_state=42)
df_high_upsampled = resample(df_high, replace=True, n_samples=len(df_major)//2, random_state=42)

dataTraining_balanced = pd.concat([df_major, df_med_upsampled, df_high_upsampled])
dataTraining_balanced = dataTraining_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

dataTraining = dataTraining_balanced

# Preparar MultiLabelBinarizers para las 5 categorías
CATEGORIAS = ['violencia', 'sexualidad', 'lenguaje_fuerte', 'drogas', 'suicidio']
NIVELES = ["sin_contenido", "moderado", "alto"]

mlb_dict = {cat: MultiLabelBinarizer(classes=NIVELES) for cat in CATEGORIAS}
Y_dict = {}

# Procesar cada categoría
for cat in CATEGORIAS:
    col_name = f'{cat}_nivel'
    if col_name in dataTraining.columns:
        niv_cat = dataTraining[col_name].astype(str).str.strip().str.lower()
        niv_cat = niv_cat.replace({"nan": "", "none": "", "": ""})
        dataTraining[f'{cat}_nivel_list'] = niv_cat.apply(
            lambda s: [s] if s in NIVELES else []
        )
        Y_dict[cat] = mlb_dict[cat].fit_transform(dataTraining[f'{cat}_nivel_list'])
        print(f"\nCategoría {cat}:")
        print("  Únicos:", sorted(set(niv_cat.unique()) - {""})) 
        print("  Distribución:", niv_cat[niv_cat != ""].value_counts().to_dict())
    else:
        print(f"\nAdvertencia: columna '{col_name}' no encontrada, usando valores vacíos")
        Y_dict[cat] = mlb_dict[cat].fit_transform([[]] * len(dataTraining))

# Combinar todas las categorías en una matriz multi-output
Y = np.column_stack([Y_dict[cat] for cat in CATEGORIAS])
print(f"\nShape final Y (multi-output): {Y.shape}")
print(f"Total de etiquetas: {Y.shape[1]} ({len(CATEGORIAS)} categorías × {len(NIVELES)} niveles)")



tfidf_word = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
)

tfidf_char = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        min_df=3,
        max_features=20_000,
        sublinear_tf=True
)

X_word = tfidf_word.fit_transform(dataTraining['clean_text'])
X_char = tfidf_char.fit_transform(dataTraining['clean_text'])

X_tfidf = hstack([X_word, X_char])



sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

X_sbert = sbert_model.encode(dataTraining['description'].astype(str).tolist(), show_progress_bar=True)



X_sbert_sparse = csr_matrix(X_sbert)

X_combined = hstack([X_tfidf, X_sbert_sparse])



try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

def get_classifier(classifier_name):
    classifiers = {
        'logistic': LogisticRegression(solver='liblinear', max_iter=1000),
        'randomforest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced', max_depth=15),
        'lightgbm': LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced', max_depth=6, learning_rate=0.1, verbose=-1) if LGBMClassifier else LogisticRegression(solver='liblinear', max_iter=1000)
    }
    return classifiers.get(classifier_name, classifiers['logistic'])

def select_classifier():
    print("\nPanel de Seleccion de Clasificadores")
    print("=" * 40)
    print("1. Logistic Regression")
    print("2. Random Forest")
    print("3. LightGBM")
    print("=" * 40)
    
    choice = input("Selecciona el clasificador (1-3): ").strip()
    
    if choice == '1':
        return 'logistic'
    elif choice == '2':
        return 'randomforest'
    elif choice == '3':
        return 'lightgbm'
    else:
        print("Opcion invalida. Usando Logistic Regression por defecto.")
        return 'logistic'

CLASSIFIER_CHOICE = select_classifier()

print(f"Usando clasificador: {CLASSIFIER_CHOICE}")

X_train, X_test, Y_train, Y_test = train_test_split(X_combined, Y, test_size=0.2, random_state=42)

# Usar MultiOutputClassifier para predicción multi-categoría
clf = MultiOutputClassifier(get_classifier(CLASSIFIER_CHOICE))
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

print("\n" + "="*80)
print("EVALUACIÓN DEL MODELO MULTI-CATEGORÍA (5 categorías de sensibilidad)")
print("="*80)

# Evaluar cada categoría por separado
offset = 0
mcauc_scores = []

for i, cat in enumerate(CATEGORIAS):
    n_labels = len(NIVELES)
    Y_test_cat = Y_test[:, offset:offset+n_labels]
    Y_pred_cat = Y_pred[:, offset:offset+n_labels]
    
    # Extraer probabilidades correctamente para esta categoría
    # Cada estimador en clf.estimators_[offset:offset+n_labels] predice una columna binaria
    y_pred_proba_cat = np.column_stack([
        clf.estimators_[offset + j].predict_proba(X_test)[:, 1] 
        for j in range(n_labels)
    ])
    
    print(f"\n{'─'*80}")
    print(f"CATEGORÍA: {cat.upper()}")
    print(f"{'─'*80}")
    print(classification_report(Y_test_cat, Y_pred_cat, target_names=NIVELES, zero_division=0))
    
    # Calcular AUC para esta categoría
    valid_cols_cat = [j for j in range(n_labels) if len(np.unique(Y_test_cat[:, j])) > 1]
    if valid_cols_cat:
        try:
            mc_auc_cat = roc_auc_score(Y_test_cat[:, valid_cols_cat], y_pred_proba_cat[:, valid_cols_cat], average='macro')
            print(f"MCAUC (macro): {round(mc_auc_cat, 4)}")
            mcauc_scores.append(mc_auc_cat)
        except Exception as e:
            print(f"MCAUC: no calculable ({e})")
    else:
        print("MCAUC: no hay clases válidas")
    
    offset += n_labels

# Evaluación global
print(f"\n{'═'*80}")
print("RESUMEN GLOBAL")
print(f"{'═'*80}")

if mcauc_scores:
    mcauc_promedio = np.mean(mcauc_scores)
    print(f"MCAUC promedio (todas las categorías): {round(mcauc_promedio, 4)}")
    print(f"Categorías evaluadas: {len(mcauc_scores)} de {len(CATEGORIAS)}")
    
    # Detalle por categoría
    print("\nMCAUC por categoría:")
    for cat, score in zip([c for c in CATEGORIAS if len(mcauc_scores) > 0], mcauc_scores):
        print(f"  - {cat}: {round(score, 4)}")
else:
    print("No se pudo calcular MCAUC para ninguna categoría")

# Guardar modelo y vectorizadores
print(f"\n{'═'*80}")
print("GUARDANDO MODELO Y VECTORIZADORES")
print(f"{'═'*80}")

import joblib
from pathlib import Path

# Crear directorio de modelos
base_dir = Path(__file__).resolve().parents[2]
models_dir = base_dir / "models" / "production"
models_dir.mkdir(parents=True, exist_ok=True)

# Guardar modelo
model_path = models_dir / "model_multicategoria.pkl"
joblib.dump(clf, model_path)
print(f" Modelo guardado en: {model_path}")

# Guardar vectorizadores TF-IDF
tfidf_word_path = models_dir / "tfidf_word.pkl"
joblib.dump(tfidf_word, tfidf_word_path)
print(f" TF-IDF Word guardado en: {tfidf_word_path}")

tfidf_char_path = models_dir / "tfidf_char.pkl"
joblib.dump(tfidf_char, tfidf_char_path)
print(f" TF-IDF Char guardado en: {tfidf_char_path}")

print(f"\n Todos los archivos guardados exitosamente en: {models_dir}")
print(f"\nAhora puede ejecutar predicciones con: python predict_movie.py")
print(f"{'═'*80}")