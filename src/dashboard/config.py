"""
Configuration settings for the dashboard application.
"""

import os

# Prediction API
API_BASE_URL = os.getenv("API_BASE_URL", "http://filmlens-api:8000")
API_PREDICT_ENDPOINT = f"{API_BASE_URL}/api/v1/predict"
API_HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# Query API
QUERY_API_BASE_URL = os.getenv("QUERY_API_BASE_URL", "http://filmlens-query-api:8001")
QUERY_API_SEARCH_ENDPOINT = f"{QUERY_API_BASE_URL}/api/v1/movies/search"
QUERY_API_AUTOCOMPLETE_ENDPOINT = f"{QUERY_API_BASE_URL}/api/v1/movies/autocomplete"
QUERY_API_FILTERS_ENDPOINT = f"{QUERY_API_BASE_URL}/api/v1/movies/filters/options"
QUERY_API_MOVIE_DETAIL_ENDPOINT = f"{QUERY_API_BASE_URL}/api/v1/movies"

DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8050"))
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Autocomplete settings
AUTOCOMPLETE_DEBOUNCE_MS = 300
AUTOCOMPLETE_MIN_CHARS = 2
AUTOCOMPLETE_LIMIT = 10

# Multilevel trigger information
TRIGGER_INFO = {
    "drogas_nivel": {
        "name": "Drogas",
        "icon": "💊",
        "descriptions": {
            "sin_contenido": "Se mencionan sustancias de forma incidental o en contextos médicos, sin mostrar consumo ni promoverlo.",
            "moderado": "Hay expresiones coloquiales o exclamaciones intensas, pero sin insultos ni lenguaje ofensivo sostenido.",
            "alto": "Presencia de insultos, expresiones ofensivas o lenguaje vulgar.",
        },
    },
    "lenguaje_fuerte_nivel": {
        "name": "Lenguaje fuerte",
        "icon": "🗣️",
        "descriptions": {
            "sin_contenido": "Hay expresiones coloquiales o exclamaciones intensas, pero sin insultos ni lenguaje ofensivo sostenido.",
            "moderado": "Hay expresiones coloquiales o exclamaciones intensas, pero sin insultos ni lenguaje ofensivo sostenido.",
            "alto": "Presencia de insultos, expresiones ofensivas o lenguaje vulgar.",
        },
    },
    "suicidio_nivel": {
        "name": "Suicidio",
        "icon": "💔",
        "descriptions": {
            "sin_contenido": "El tema se menciona de forma indirecta o simbólica, sin representación emocional ni visual.",
            "moderado": "Menciones, representaciones emocionales o escenas explícitas relacionadas con el tema.",
            "alto": "Menciones, representaciones emocionales o escenas explícitas relacionadas con el tema.",
        },
    },
    "sexualidad_nivel": {
        "name": "Sexualidad",
        "icon": "❤️",
        "descriptions": {
            "sin_contenido": "El tema se menciona de forma indirecta o simbólica, sin representación emocional ni visual.",
            "moderado": "Escenas íntimas, contacto físico o situaciones que involucran deseo, consentimiento o exposición.",
            "alto": "Escenas íntimas, contacto físico o situaciones que involucran deseo, consentimiento o exposición.",
        },
    },
    "violencia_nivel": {
        "name": "Violencia",
        "icon": "🔪",
        "descriptions": {
            "sin_contenido": "El tema se menciona de forma indirecta o simbólica, sin representación emocional ni visual.",
            "moderado": "Conflictos físicos, agresiones, daño gráfico o contacto visual intensa.",
            "alto": "Conflictos físicos, agresiones, daño gráfico o violencia visual intensa.",
        },
    },
}

# UI Constants
UI_CONSTANTS = {
    # Autocomplete styles
    "AUTOCOMPLETE_HOVER_COLOR": "#f8f9fa",
    "AUTOCOMPLETE_TRANSITION_MS": "0.2s",
    "AUTOCOMPLETE_MAX_HEIGHT": "300px",
    "AUTOCOMPLETE_DROPDOWN_OFFSET": "5px",
    "AUTOCOMPLETE_SHADOW": "0 4px 12px rgba(0,0,0,0.15)",
    # Font sizes
    "FONT_SIZE_ICON_LARGE": "48px",
    "FONT_SIZE_ICON_MEDIUM": "32px",
    "FONT_SIZE_ICON_SMALL": "20px",
    "FONT_SIZE_TEXT_SMALL": "12px",
    "FONT_SIZE_TEXT_NORMAL": "14px",
    "FONT_SIZE_TEXT_MEDIUM": "16px",
    "FONT_SIZE_TEXT_LARGE": "18px",
    # Spacing
    "PADDING_SMALL": "8px 12px",
    "PADDING_MEDIUM": "12px",
    "PADDING_LARGE": "20px 15px",
    "MARGIN_SMALL": "5px",
    "MARGIN_MEDIUM": "15px",
    # Component heights
    "TRIGGER_CARD_DESCRIPTION_MIN_HEIGHT": "55px",
    "INPUT_PADDING_LEFT": "45px",
    # Colors
    "COLOR_TEXT_MUTED": "#999",
    "COLOR_TEXT_GRAY": "#666",
    "COLOR_TEXT_DARK": "#333",
    "COLOR_BORDER_LIGHT": "#e0e0e0",
    "COLOR_BORDER": "#ddd",
    "COLOR_WHITE": "white",
    "COLOR_BACKGROUND_LIGHT": "#f0f0f0",
    # Border radius
    "BORDER_RADIUS_SMALL": "8px",
    "BORDER_RADIUS_MEDIUM": "20px",
}
