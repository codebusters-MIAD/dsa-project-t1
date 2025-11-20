"""
Feature engineering para el modelo de sensibilidad.
Incluye limpieza de texto, TF-IDF y embeddings SBERT.
"""

import re
import nltk
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_tokenize(text: str) -> list:
    """
    Limpia y tokeniza texto: lowercase, remove special chars, stopwords, lemmatization.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return words


def prepare_text_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Prepara features de texto: limpieza, tokenizacion y longitud.
    """
    text_column = config['data_config']['text_column']
    min_length = config['data_config']['min_plot_length']
    
    df['clean_tokens'] = df[text_column].apply(clean_tokenize)
    df['clean_text'] = df['clean_tokens'].apply(lambda x: ' '.join(x))
    df['plot_length'] = df['clean_tokens'].apply(len)
    
    df = df[df['plot_length'] >= min_length].reset_index(drop=True)
    
    return df


def create_tfidf_features(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict) -> Tuple:
    """
    Crea features TF-IDF (word-level y character-level) para train y test.
    Retorna vectorizadores y matrices sparse.
    """
    word_cfg = config['feature_config']['tfidf_word']
    char_cfg = config['feature_config']['tfidf_char']
    
    tfidf_word = TfidfVectorizer(
        max_features=word_cfg['max_features'],
        ngram_range=tuple(word_cfg['ngram_range']),
        min_df=word_cfg['min_df'],
        max_df=word_cfg['max_df'],
        sublinear_tf=word_cfg['sublinear_tf']
    )
    
    tfidf_char = TfidfVectorizer(
        analyzer=char_cfg['analyzer'],
        ngram_range=tuple(char_cfg['ngram_range']),
        min_df=char_cfg['min_df'],
        max_features=char_cfg['max_features'],
        sublinear_tf=char_cfg['sublinear_tf']
    )
    
    X_train_word = tfidf_word.fit_transform(train_df['clean_text'])
    X_test_word = tfidf_word.transform(test_df['clean_text'])
    
    X_train_char = tfidf_char.fit_transform(train_df['clean_text'])
    X_test_char = tfidf_char.transform(test_df['clean_text'])
    
    X_train_tfidf = hstack([X_train_word, X_train_char])
    X_test_tfidf = hstack([X_test_word, X_test_char])
    
    return X_train_tfidf, X_test_tfidf, tfidf_word, tfidf_char


def create_sbert_features(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict) -> Tuple:
    """
    Crea embeddings SBERT para train y test.
    Retorna matrices sparse y el modelo SBERT.
    """
    if not config['feature_config']['sbert']['use_sbert']:
        return None, None, None
    
    model_name = config['feature_config']['sbert']['model_name']
    text_column = config['data_config']['text_column']
    
    sbert_model = SentenceTransformer(model_name)
    
    X_train_sbert = sbert_model.encode(
        train_df[text_column].astype(str).tolist(),
        show_progress_bar=False
    )
    
    X_test_sbert = sbert_model.encode(
        test_df[text_column].astype(str).tolist(),
        show_progress_bar=False
    )
    
    X_train_sbert_sparse = csr_matrix(X_train_sbert)
    X_test_sbert_sparse = csr_matrix(X_test_sbert)
    
    return X_train_sbert_sparse, X_test_sbert_sparse, sbert_model


def combine_features(X_tfidf, X_sbert=None):
    """
    Combina features TF-IDF y SBERT (opcional).
    """
    if X_sbert is None:
        return X_tfidf
    return hstack([X_tfidf, X_sbert])


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict):
    """
    Pipeline completo de feature engineering: TF-IDF + SBERT.
    Retorna matrices de features y objetos de transformacion.
    """
    train_df = prepare_text_features(train_df, config)
    test_df = prepare_text_features(test_df, config)
    
    X_train_tfidf, X_test_tfidf, tfidf_word, tfidf_char = create_tfidf_features(
        train_df, test_df, config
    )
    
    X_train_sbert, X_test_sbert, sbert_model = create_sbert_features(
        train_df, test_df, config
    )
    
    X_train = combine_features(X_train_tfidf, X_train_sbert)
    X_test = combine_features(X_test_tfidf, X_test_sbert)
    
    transformers = {
        'tfidf_word': tfidf_word,
        'tfidf_char': tfidf_char,
        'sbert_model': sbert_model
    }
    
    return X_train, X_test, train_df, test_df, transformers
