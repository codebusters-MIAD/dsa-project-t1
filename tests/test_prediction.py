"""Unit tests for prediction module."""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from filmlens.predict import make_prediction


def test_make_prediction_with_dict():
    """Test prediction with dictionary input."""
    try:
        input_data = {
            'description': 'A violent action movie with guns and explosions'
        }
        
        result = make_prediction(input_data)
        
        assert 'results' in result
        assert 'version' in result
        assert len(result['results']) == 1
        assert 'predictions' in result['results'][0]
        
    except FileNotFoundError:
        pytest.skip("Model not trained yet")


def test_make_prediction_with_dataframe():
    """Test prediction with DataFrame input."""
    try:
        df = pd.DataFrame({
            'description': [
                'A violent action film',
                'A romantic comedy'
            ]
        })
        
        result = make_prediction(df)
        
        assert len(result['results']) == 2
        
    except FileNotFoundError:
        pytest.skip("Model not trained yet")
