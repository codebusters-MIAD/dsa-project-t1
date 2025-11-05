"""Unit tests for data manager."""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from filmlens.processing.data_manager import (
    load_dataset,
    pre_pipeline_preparation
)


def test_load_dataset():
    """Test dataset loading."""
    try:
        df = load_dataset()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    except FileNotFoundError:
        pytest.skip("Dataset not found")


def test_pre_pipeline_preparation():
    """Test data preparation."""
    df = pd.DataFrame({
        'description': ['A violent movie', 'A romantic comedy', None],
        'title': ['Movie 1', 'Movie 2', 'Movie 3']
    })
    
    df_clean = pre_pipeline_preparation(df)
    
    # Should drop null descriptions
    assert len(df_clean) == 2
    
    # Should add basic features
    assert 'description_length' in df_clean.columns
    assert 'word_count' in df_clean.columns
