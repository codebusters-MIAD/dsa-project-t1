import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import chain

dataTraining = df = pd.read_csv("../../data/processed/ml/dataset_sensibilidad_imdb_final_complete.csv")

dataTraining.info()

from scipy.sparse import hstack, csr_matrix

import pandas as pd
import numpy as np
import ast
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

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

NIVELES_VALIDOS = {"leve", "medio", "alto"}

niv = dataTraining['violencia_nivel'].astype(str).str.strip().str.lower()

niv = niv.replace({"nan": "", "none": "", "": ""})

dataTraining['violencia_nivel_list'] = niv.apply(
    lambda s: [s] if s in NIVELES_VALIDOS else []
)

from sklearn.utils import resample

df_major = dataTraining[dataTraining['violencia_nivel'] == 'leve']
df_med = dataTraining[dataTraining['violencia_nivel'] == 'medio']
df_high = dataTraining[dataTraining['violencia_nivel'] == 'alto']

df_med_upsampled = resample(df_med, replace=True, n_samples=len(df_major)//2, random_state=42)
df_high_upsampled = resample(df_high, replace=True, n_samples=len(df_major)//2, random_state=42)

dataTraining_balanced = pd.concat([df_major, df_med_upsampled, df_high_upsampled])
dataTraining_balanced = dataTraining_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=["leve", "medio", "alto"])
Y = mlb.fit_transform(dataTraining['violencia_nivel_list'])

print("Únicos en violencia_nivel:", sorted(set(niv.unique()) - {""}))
print("Distribución:", niv[niv != ""].value_counts())

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

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

from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

X_sbert = sbert_model.encode(dataTraining['description'].astype(str).tolist(), show_progress_bar=True)

from scipy.sparse import csr_matrix

X_sbert_sparse = csr_matrix(X_sbert)

X_combined = hstack([X_tfidf, X_sbert_sparse])

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

X_train, X_test, Y_train, Y_test = train_test_split(X_combined, Y, test_size=0.2, random_state=42)

clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000))
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

print("Evaluación del modelo híbrido TF-IDF:")
print(classification_report(Y_test, Y_pred, target_names=mlb.classes_))

valid_cols = [i for i in range(Y_test.shape[1]) if len(np.unique(Y_test[:, i])) > 1]

if valid_cols:
    mc_auc = roc_auc_score(Y_test[:, valid_cols], y_pred_proba[:, valid_cols], average='macro')
    print(f"MCAUC (Mean Column-wise AUC, macro) sobre {len(valid_cols)} clases válidas de {Y_test.shape[1]}:", round(mc_auc, 4))
else:
    print("No hay clases válidas para calcular el AUC.")