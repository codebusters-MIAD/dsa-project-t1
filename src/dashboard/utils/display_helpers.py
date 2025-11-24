"""
Helper functions for displaying prediction results.
"""

from typing import Dict, List
from dash import html
import dash_bootstrap_components as dbc

from config import TRIGGER_INFO
from components import create_trigger_card_multilevel


def create_movie_info_section(movie: Dict) -> html.Div:
    """
    Create movie information display section.

    Args:
        movie: Dictionary containing movie details

    Returns:
        html.Div with formatted movie information
    """
    genres = ", ".join(movie.get("genre", [])[:3]) if movie.get("genre") else "N/A"

    return html.Div(
        [
            html.H3(
                [html.Span("🎬 ", style={"fontSize": "32px"}), movie.get("movie_name", "Unknown")], className="mb-2"
            ),
            html.Div(
                [
                    dbc.Badge(f"📅 {movie.get('year', 'N/A')}", color="info", className="me-2"),
                    dbc.Badge(f"⏱️ {movie.get('runtime', 'N/A')} min", color="secondary", className="me-2"),
                    dbc.Badge(f"⭐ {movie.get('rating', 'N/A')}", color="warning"),
                ],
                className="mb-3",
            ),
            html.P([html.Strong("Género: "), html.Span(genres)], className="mb-2"),
            html.P(movie.get("description", ""), className="text-muted", style={"fontSize": "14px"}),
        ]
    )


def create_suitability_message(score: float) -> str:
    """
    Generate family suitability message based on score.

    Args:
        score: Family suitability score (0-100)

    Returns:
        Descriptive message about movie suitability
    """
    if score >= 70:
        return "La película es segura para ver en familia. Tiene bajo contenido sensible y es adecuada para niños entre 7 y 12 años."
    elif score >= 40:
        return "La película tiene contenido moderadamente sensible. Revisa los detalles abajo antes de decidir si es apropiada para tu familia."
    else:
        return "La película contiene contenido sensible significativo. Considera cuidadosamente si es apropiada para niños entre 7 y 12 años."


def create_predictions_dict(prediction: Dict) -> Dict:
    """
    Extract predictions dictionary from prediction response.

    Args:
        prediction: Full prediction response

    Returns:
        Dictionary with categorized predictions
    """
    return {
        "violencia_nivel": prediction.get("violencia_nivel", {}),
        "sexualidad_nivel": prediction.get("sexualidad_nivel", {}),
        "drogas_nivel": prediction.get("drogas_nivel", {}),
        "lenguaje_fuerte_nivel": prediction.get("lenguaje_fuerte_nivel", {}),
        "suicidio_nivel": prediction.get("suicidio_nivel", {}),
    }


def create_trigger_cards_layout(predictions_dict: Dict) -> dbc.Row:
    """
    Create trigger cards grid layout.

    Args:
        predictions_dict: Dictionary with predictions for each trigger category

    Returns:
        dbc.Row with trigger cards in responsive grid
    """
    trigger_cards = []

    for trigger_key, trigger_data in predictions_dict.items():
        if trigger_key in TRIGGER_INFO:
            info = TRIGGER_INFO[trigger_key]
            nivel = trigger_data.get("nivel", "sin_contenido")
            probability = trigger_data.get("probabilidad", 0.0)
            probabilities_all = trigger_data.get("probabilidades_todas", {})
            description = info["descriptions"].get(nivel, "")

            card = create_trigger_card_multilevel(
                trigger_name=trigger_key,
                spanish_name=info["name"],
                icon=info["icon"],
                nivel=nivel,
                probability=probability,
                probabilities_all=probabilities_all,
                description=description,
            )
            trigger_cards.append(card)

    # Wrap trigger cards in a responsive grid layout (5 columns)
    trigger_cols = [dbc.Col(card, xs=12, sm=6, md=4, lg=2, className="mb-3") for card in trigger_cards]

    return dbc.Row(trigger_cols, className="g-3")
