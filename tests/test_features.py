"""Unit tests for feature transformers."""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from filmlens.processing.features import (
    TextCleaner,
    KeywordDetector,
    GenreEncoder
)


def test_text_cleaner():
    """Test text cleaning transformer."""
    df = pd.DataFrame({
        'description': ['A VIOLENT Movie!!! 123', 'A romantic comedy.']
    })
    
    cleaner = TextCleaner(text_column='description')
    df_clean = cleaner.fit_transform(df)
    
    assert 'description_clean' in df_clean.columns
    assert 'violent movie' in df_clean['description_clean'].iloc[0]
    assert '123' not in df_clean['description_clean'].iloc[0]
    assert '!' not in df_clean['description_clean'].iloc[0]


def test_keyword_detector():
    """Test keyword detection transformer."""
    df = pd.DataFrame({
        'description_clean': [
            'a violent action movie with guns',
            'a romantic comedy about love'
        ]
    })
    
    detector = KeywordDetector(
        text_column='description_clean',
        categories=['violence']
    )
    df_feat = detector.fit_transform(df)
    
    assert 'has_violence' in df_feat.columns
    assert df_feat['has_violence'].iloc[0] == 1
    assert df_feat['has_violence'].iloc[1] == 0


def test_genre_encoder():
    """Test genre encoding transformer."""
    df_train = pd.DataFrame({
        'genre': ['action', 'comedy', 'action']
    })
    
    df_test = pd.DataFrame({
        'genre': ['action', 'drama']
    })
    
    encoder = GenreEncoder()
    encoder.fit(df_train)
    df_encoded = encoder.transform(df_test)
    
    assert 'genre_action' in df_encoded.columns
    assert 'genre_comedy' in df_encoded.columns
