import pandas as pd
import numpy as np
import re
import string
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import logging

from filmlens.config import config

logger = logging.getLogger(__name__)


class TextCleaner(BaseEstimator, TransformerMixin):
    """Clean and normalize text."""
    
    def __init__(self, text_column: str = None):
        if text_column is None:
            text_column = config.data_config.text_column
        self.text_column = text_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        def clean_text(text):
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = re.sub(r'\d+', '', text)
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = ' '.join(text.split())
            return text
        
        X[f'{self.text_column}_clean'] = X[self.text_column].apply(clean_text)
        return X


class KeywordDetector(BaseEstimator, TransformerMixin):
    """
    Extract binary keyword features for triggers.
    """
    
    KEYWORDS = {
        'violence': ['violence', 'violent', 'kill', 'murder', 'death', 'blood', 
                     'fight', 'war', 'weapon', 'gun', 'shooting', 'stabbing'],
        'sexual_content': ['sex', 'sexual', 'rape', 'assault', 'seduce', 
                          'affair', 'prostitut', 'harassment'],
        'substance_abuse': ['drug', 'cocaine', 'heroin', 'addiction', 'alcohol', 
                           'drunk', 'overdose', 'addict', 'substance'],
        'suicide': ['suicide', 'suicidal', 'kill himself', 'kill herself', 
                   'depression', 'self-harm'],
        'child_abuse': ['child abuse', 'pedophil', 'underage'],
        'discrimination': ['racism', 'racist', 'discrimination', 'prejudice', 
                          'hate crime', 'sexism', 'homophobia'],
        'strong_language': ['profanity', 'vulgar', 'explicit language', 'offensive'],
        'horror': ['horror', 'terror', 'scary', 'frightening', 'nightmare', 
                  'haunted', 'ghost', 'demon'],
        'animal_cruelty': ['animal cruelty', 'animal abuse', 'torture animal']
    }
    
    def __init__(self, text_column: str = None, categories: List[str] = None, 
                 use_as_features: bool = False):
        """
        Args:
            text_column: Column to search for keywords
            categories: List of trigger categories to detect
            use_as_features: If True, creates keyword_* columns as features.
                            If False (default), doesn't create any columns.
        """
        if text_column is None:
            text_column = config.data_config.text_column
        if categories is None:
            categories = config.feature_config.keyword_categories
        
        self.text_column = text_column
        self.categories = categories
        self.use_as_features = use_as_features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Only create keyword features if explicitly requested
        if not self.use_as_features:
            return X
        
        for category in self.categories:
            if category not in self.KEYWORDS:
                continue
            
            keywords = self.KEYWORDS[category]
            pattern = '|'.join(keywords)
            
            # Use keyword_ prefix to distinguish from target labels
            X[f'keyword_{category}'] = X[self.text_column].str.contains(
                pattern, case=False, na=False, regex=True
            ).astype(int)
        
        return X


def create_keyword_labels(df: pd.DataFrame, text_column: str = None) -> pd.DataFrame:
    """
    Create target labels based on keywords (for baseline only).
    
    This is used to create initial labels when no ground truth exists.
    Returns DataFrame with has_* columns.
    """
    if text_column is None:
        text_column = config.data_config.text_column
    
    labels_df = df.copy()
    
    for category, keywords in KeywordDetector.KEYWORDS.items():
        pattern = '|'.join(keywords)
        labels_df[f'has_{category}'] = labels_df[text_column].str.contains(
            pattern, case=False, na=False, regex=True
        ).astype(int)
    
    return labels_df


class GenreEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode genre column."""
    
    def __init__(self, genre_column: str = 'genre'):
        self.genre_column = genre_column
        self.genre_columns_ = None
    
    def fit(self, X, y=None):
        if self.genre_column in X.columns:
            genre_dummies = pd.get_dummies(X[self.genre_column], prefix='genre')
            self.genre_columns_ = genre_dummies.columns.tolist()
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if self.genre_column not in X.columns:
            return X
        
        genre_dummies = pd.get_dummies(X[self.genre_column], prefix='genre')
        
        # Align with training columns
        if self.genre_columns_:
            for col in self.genre_columns_:
                if col not in genre_dummies.columns:
                    genre_dummies[col] = 0
            genre_dummies = genre_dummies[self.genre_columns_]
        
        X = pd.concat([X, genre_dummies], axis=1)
        return X


class FeatureCombiner(BaseEstimator, TransformerMixin):
    """Combine TF-IDF, keyword, and genre features."""
    
    def __init__(self, text_column: str = None):
        if text_column is None:
            text_column = f"{config.data_config.text_column}_clean"
        
        self.text_column = text_column
        self.tfidf = None
        self.genre_columns = None
        self.keyword_columns = None
        
        # Initialize TF-IDF from config
        tfidf_cfg = config.feature_config.tfidf
        self.tfidf = TfidfVectorizer(
            max_features=tfidf_cfg.max_features,
            ngram_range=tuple(tfidf_cfg.ngram_range),
            min_df=tfidf_cfg.min_df,
            max_df=tfidf_cfg.max_df,
            stop_words=tfidf_cfg.stop_words
        )
    
    def fit(self, X, y=None):
        # Fit TF-IDF
        self.tfidf.fit(X[self.text_column])
        
        # Store genre columns
        if config.feature_config.use_genre_features:
            self.genre_columns = [col for col in X.columns if col.startswith('genre_')]
        
        # Store keyword columns
        if config.feature_config.use_keyword_features:
            self.keyword_columns = [col for col in X.columns if col.startswith('keyword_')]
        
        return self
    
    def transform(self, X):
        # TF-IDF features
        X_tfidf = self.tfidf.transform(X[self.text_column])
        
        features_list = [X_tfidf]
        
        # Add genre features
        if self.genre_columns:
            X_genre = csr_matrix(X[self.genre_columns].values)
            features_list.append(X_genre)
        
        # Add keyword features
        if self.keyword_columns:
            X_keywords = csr_matrix(X[self.keyword_columns].values)
            features_list.append(X_keywords)
        
        # Combine all features
        X_combined = hstack(features_list)
        return X_combined
