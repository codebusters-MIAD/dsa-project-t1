"""
Configuration settings for the dashboard application.
"""

import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://filmlens-api:8000")
API_PREDICT_ENDPOINT = f"{API_BASE_URL}/api/v1/predict"
API_HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8050"))
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

TRIGGER_LABELS = {
    'has_suicide': 'Suicidio',
    'has_substance_abuse': 'Drogas',
    'has_strong_language': 'Lenguaje fuerte',
    'has_sexual_content': 'Sexualidad',
    'has_violence': 'Violencia'
}

TRIGGER_COLORS = {
    'has_suicide': '#e74c3c',
    'has_substance_abuse': '#f39c12',
    'has_strong_language': '#e67e22',
    'has_sexual_content': '#e91e63',
    'has_violence': '#c0392b'
}

SENSITIVITY_THRESHOLDS = {
    'low': (0, 0.3),
    'medium': (0.3, 0.6),
    'high': (0.6, 1.0)
}
