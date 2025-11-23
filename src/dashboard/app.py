import logging
from typing import List, Dict, Optional, Tuple, Any
from dash import Dash, html, dcc, Input, Output, State, ALL, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from components import (
    create_search_interface,
    create_autocomplete_item,
    create_results_display,
    create_sensitivity_gauge,
)
from services import APIClient
from config import DASHBOARD_HOST, DASHBOARD_PORT, DEBUG_MODE, AUTOCOMPLETE_MIN_CHARS
from utils.calculations import calculate_family_suitability_score_multilevel, get_suitability_color
from utils.display_helpers import (
    create_movie_info_section,
    create_suitability_message,
    create_predictions_dict,
    create_trigger_cards_layout,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css",
    ],
    suppress_callback_exceptions=True,
    title="FilmLens - Descubre peliculas en familia",
)

# Add custom CSS for hover effects
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .autocomplete-item:hover {
                background-color: #f8f9fa !important;
                cursor: pointer;
            }
            .autocomplete-item {
                transition: background-color 0.2s;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

server = app.server

api_client = APIClient()


def create_layout():
    """Create the main dashboard layout."""

    return dbc.Container(
        [
            # Alert container
            dbc.Row([dbc.Col([html.Div(id="alert-container")])]),
            # Search interface
            dbc.Row([dbc.Col([create_search_interface()], md=12)]),
            # Results display
            dbc.Row([dbc.Col([create_results_display()], md=12)]),
            # Hidden div to store selected movie
            html.Div(id="selected-movie-store", style={"display": "none"}),
            # Cache store for predictions
            dcc.Store(id="predictions-cache", storage_type="session", data={}),
            # Store for prediction results
            dcc.Store(id="prediction-results-store", data=None),
        ],
        fluid=True,
        className="py-4",
        style={"maxWidth": "1200px"},
    )


app.layout = create_layout()


# Callback 2: Autocomplete suggestions
@app.callback(
    [Output("autocomplete-dropdown", "children"), Output("autocomplete-dropdown", "style")],
    [Input("search-input", "value")],
)
def update_autocomplete(search_value: Optional[str]) -> Tuple[List, Dict[str, str]]:
    """Update autocomplete dropdown with movie suggestions."""
    
    # Hide dropdown if empty or too short
    if not search_value or len(search_value) < AUTOCOMPLETE_MIN_CHARS:
        return [], {"display": "none"}
    
    logger.info(f"Autocomplete search: '{search_value}'")
    
    try:
        suggestions = api_client.autocomplete_movies(search_value.strip(), limit=10)
        
        if not suggestions:
            logger.warning(f"No suggestions found for: '{search_value}'")
            return [], {"display": "none"}
        
        logger.info(f"Found {len(suggestions)} suggestions for '{search_value}'")
        items = [create_autocomplete_item(movie) for movie in suggestions]
        
        # Wrap in ListGroup
        list_group = dbc.ListGroup(items, flush=True)
        
        return [list_group], {
            "display": "block",
            "position": "absolute",
            "top": "calc(100% + 5px)",
            "left": "0",
            "right": "0",
            "backgroundColor": "white",
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "boxShadow": "0 4px 12px rgba(0,0,0,0.15)",
            "maxHeight": "300px",
            "overflowY": "auto",
            "zIndex": "1000",
        }
    except Exception as e:
        logger.error(f"Error in autocomplete: {str(e)}")
        return [], {"display": "none"}


# Callback 3: Handle movie selection from autocomplete (only fill input and store movie_id)
@app.callback(
    [
        Output("search-input", "value"),
        Output("autocomplete-dropdown", "style", allow_duplicate=True),
        Output("autocomplete-dropdown", "children", allow_duplicate=True),
        Output("selected-movie-id", "children"),
        Output("btn-predict", "disabled"),
    ],
    [Input({"type": "autocomplete-item", "index": ALL}, "n_clicks")],
    [State({"type": "autocomplete-item", "index": ALL}, "id")],
    prevent_initial_call='initial_duplicate',
)
def handle_movie_selection(n_clicks: List[Optional[int]], ids: List[Dict[str, Any]]) -> Tuple[str, Dict[str, str], List, str, bool]:
    """Handle movie selection from autocomplete - just fill input and enable button."""
    
    # Check if callback was triggered by a click
    if not ctx.triggered or not ctx.triggered_id:
        raise PreventUpdate
    
    if not n_clicks or not any(n_clicks):
        raise PreventUpdate
    
    # Find which movie was clicked
    clicked_idx = next((i for i, clicks in enumerate(n_clicks) if clicks), None)
    if clicked_idx is None:
        raise PreventUpdate
    
    movie_id = ids[clicked_idx]["index"]
    logger.info(f"Movie selected: {movie_id}")
    
    # Get movie details to show name in input
    movie = api_client.get_movie_detail(movie_id)
    if not movie:
        logger.error(f"Failed to get movie details for: {movie_id}")
        return "", {"display": "none"}, [], "", True
    
    movie_name = movie.get("movie_name", "")
    logger.info(f"Movie details retrieved: {movie_name}")
    
    # Return: movie name in input, hide dropdown, store movie_id, enable button
    return movie_name, {"display": "none"}, [], movie_id, False


# Callback 4: Handle prediction button click
@app.callback(
    [
        Output("alert-container", "children"),
        Output("predictions-cache", "data"),
        Output("prediction-results-store", "data"),
        Output("btn-predict", "children"),
    ],
    [Input("btn-predict", "n_clicks")],
    [
        State("selected-movie-id", "children"),
        State("predictions-cache", "data"),
    ],
    prevent_initial_call=True,
)
def handle_prediction(n_clicks: Optional[int], movie_id: Optional[str], cache: Optional[Dict]) -> Tuple[Any, Dict, Optional[Dict], List]:
    """Handle prediction button click - call API and show results."""
    
    if not n_clicks or not movie_id:
        raise PreventUpdate
    
    # Initialize cache if None
    if cache is None:
        cache = {}
    
    logger.info(f"Prediction requested for movie: {movie_id}")
    
    # Button text to restore
    button_text = [
        html.I(className="bi bi-bar-chart-fill me-2"),
        "Predecir Sensibilidad Parental",
    ]
    
    # Check if prediction already exists in cache
    if movie_id in cache and "prediction" in cache[movie_id]:
        logger.info(f"Prediction found in cache for: {movie_id}")
        cached_data = cache[movie_id]
        alert = dbc.Alert(
            [
                html.H5("Análisis Completado (desde caché)", className="alert-heading"),
                html.P(f"Película: {cached_data.get('movie_name', 'Unknown')} ({cached_data.get('year', 'N/A')})"),
                html.Hr(),
                html.P("Esta película ya fue analizada previamente. Mostrando resultados del caché.", className="mb-0"),
            ],
            color="info",
            dismissable=True,
            duration=6000,
        )
        # Return with cached movie details and prediction
        results = {
            "movie": cached_data.get("movie_details", {}),
            "prediction": cached_data.get("prediction", {})
        }
        return alert, cache, results, button_text
    
    # Get movie details
    movie = api_client.get_movie_detail(movie_id)
    if not movie:
        alert = dbc.Alert("Error al obtener detalles de la película", color="danger", dismissable=True, duration=4000)
        return alert, cache, None, button_text
    
    logger.info(f"Calling prediction API for: {movie.get('movie_name')}")
    
    # Call prediction API
    genre_str = movie.get("genre", ["Unknown"])[0] if movie.get("genre") else "Unknown"
    
    result = api_client.predict(
        movie_id=movie.get("imdb_id", movie_id),
        title=movie.get("movie_name", "Unknown"),
        description=movie.get("description", ""),
        genre=genre_str,
    )
    
    if result:
        logger.info(f"Prediction successful for: {movie.get('movie_name')}")
        
        # Store in cache with prediction results
        cache[movie_id] = {
            "movie_name": movie.get("movie_name", "Unknown"),
            "year": movie.get("year", "N/A"),
            "predicted_at": "now",
            "prediction": result,
            "movie_details": movie
        }
        
        alert = dbc.Alert(
            [
                html.H5("Análisis Completado", className="alert-heading"),
                html.P(f"Película: {movie.get('movie_name', 'Unknown')} ({movie.get('year', 'N/A')})"),
                html.Hr(),
                html.P("La predicción se ha guardado exitosamente en la base de datos.", className="mb-0"),
            ],
            color="success",
            dismissable=True,
            duration=8000,
        )
    else:
        logger.error(f"Prediction failed for: {movie.get('movie_name')}")
        alert = dbc.Alert("Error al analizar la película", color="danger", dismissable=True, duration=4000)
        return alert, cache, None, button_text
    
    # Return movie + prediction for display
    results = {"movie": movie, "prediction": result}
    return alert, cache, results, button_text


# Callback 5: Display prediction results
@app.callback(
    [
        Output("results-container", "style"),
        Output("movie-info-display", "children"),
        Output("sensitivity-gauge", "figure"),
        Output("family-suitability-text", "children"),
        Output("triggers-display", "children"),
    ],
    [Input("prediction-results-store", "data")],
    prevent_initial_call=True,
)
def display_results(results_data: Optional[Dict]) -> Tuple[Dict[str, str], Optional[Any], Dict, Optional[str], Optional[Any]]:
    """Display prediction results with movie info, gauge, and trigger cards."""
    
    if not results_data or "movie" not in results_data or "prediction" not in results_data:
        # Hide results container
        return {"display": "none"}, None, {}, None, None
    
    movie = results_data["movie"]
    prediction = results_data["prediction"]
    
    logger.info(f"Displaying results for: {movie.get('movie_name')}")
    
    # Create movie info section
    movie_info = create_movie_info_section(movie)
    
    # Extract predictions and calculate score
    predictions_dict = create_predictions_dict(prediction)
    score = calculate_family_suitability_score_multilevel(predictions_dict)
    color = get_suitability_color(score)
    
    # Create gauge
    gauge_fig = create_sensitivity_gauge(score, color)
    
    # Generate suitability message
    suitability_text = create_suitability_message(score)
    
    # Create trigger cards layout
    trigger_layout = create_trigger_cards_layout(predictions_dict)
    
    # Show results container
    return {"display": "block"}, movie_info, gauge_fig, suitability_text, trigger_layout


def run_server():
    """Start the dashboard server."""

    logger.info("Starting FilmLens Dashboard...")
    logger.info(f"Checking API connectivity...")

    if api_client.check_health():
        logger.info("API connection successful")
    else:
        logger.warning("Unable to connect to API - predictions may fail")

    logger.info(f"Dashboard running on http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")

    app.run_server(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DEBUG_MODE)


if __name__ == "__main__":
    run_server()
