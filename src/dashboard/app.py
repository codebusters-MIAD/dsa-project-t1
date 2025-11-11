import logging
from dash import Dash, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

from components import create_input_form, create_results_display, create_sensitivity_gauge, create_trigger_card
from services import APIClient
from utils import calculate_family_suitability_score, get_suitability_level, get_suitability_color
from config import DASHBOARD_HOST, DASHBOARD_PORT, DEBUG_MODE, TRIGGER_LABELS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="FilmLens - Analisis de Sensibilidad"
)

server = app.server

api_client = APIClient()


def create_layout():
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1("FilmLens Dashboard", className="display-4 mb-2"),
                    html.P(
                        "Analisis de contenido sensible en peliculas",
                        className="lead text-muted"
                    ),
                ], className="text-center my-4")
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div(id="alert-container")
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                create_input_form()
            ], md=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                create_results_display()
            ], md=12)
        ])
        
    ], fluid=False, className="py-4")


app.layout = create_layout()


@app.callback(
    [
        Output("results-container", "style"),
        Output("sensitivity-gauge", "figure"),
        Output("movie-info-display", "children"),
        Output("triggers-display", "children"),
        Output("family-suitability-text", "children"),
        Output("alert-container", "children")
    ],
    [Input("btn-analyze", "n_clicks")],
    [
        State("input-movie-id", "value"),
        State("input-title", "value"),
        State("input-description", "value"),
        State("input-genre", "value")
    ],
    prevent_initial_call=True
)
def analyze_movie(n_clicks, movie_id, title, description, genre):    
    
    if not all([movie_id, title, description, genre]):
        alert = dbc.Alert(
            "Por favor complete todos los campos del formulario",
            color="warning",
            dismissable=True,
            duration=4000
        )
        return {'display': 'none'}, {}, None, None, None, alert
    
    if len(description) < 10:
        alert = dbc.Alert(
            "La descripcion debe tener al menos 10 caracteres",
            color="warning",
            dismissable=True,
            duration=4000
        )
        return {'display': 'none'}, {}, None, None, None, alert
    
    result = api_client.predict(
        movie_id=movie_id.strip(),
        title=title.strip(),
        description=description.strip(),
        genre=genre
    )
    
    if result is None:
        alert = dbc.Alert(
            "Error al conectar con el servicio de predicciones. Por favor intente nuevamente.",
            color="danger",
            dismissable=True,
            duration=4000
        )
        return {'display': 'none'}, {}, None, None, None, alert
    
    predictions = result.get('predictions', [])
    
    suitability_score = calculate_family_suitability_score(predictions)
    suitability_level = get_suitability_level(suitability_score)
    gauge_color = get_suitability_color(suitability_score)
    
    gauge_fig = create_sensitivity_gauge(suitability_score, gauge_color)
    
    movie_info = html.Div([
        html.H3(title, className="mb-2"),
        html.P([
            html.Strong("ID: "),
            html.Span(movie_id, className="text-muted me-3"),
            html.Strong("Genero: "),
            html.Span(genre, className="text-muted")
        ]),
        html.P(description, className="text-muted", style={'fontSize': '14px'})
    ])
    
    trigger_cards = []
    for pred in predictions:
        trigger_key = pred['trigger']
        spanish_name = TRIGGER_LABELS.get(trigger_key, trigger_key)
        
        from config import TRIGGER_COLORS
        color = TRIGGER_COLORS.get(trigger_key, '#95a5a6')
        
        card = create_trigger_card(
            trigger_key,
            spanish_name,
            pred['detected'],
            pred['probability'],
            color
        )
        trigger_cards.append(card)
    
    suitability_messages = {
        'alto': "La pelicula es muy apta para ver en familia. Tiene bajo contenido sensible y es adecuada para ninos entre 7 y 12 anos.",
        'medio': "La pelicula es moderadamente apta para familia. Tiene contenido sensible moderado. Se recomienda supervision parental para menores de 13 anos.",
        'bajo': "La pelicula es poco apta para ver en familia. Contiene contenido sensible significativo. No es recomendada para menores de 16 anos sin supervision."
    }
    
    suitability_text = suitability_messages.get(suitability_level, "")
    
    success_alert = dbc.Alert(
        "Analisis completado exitosamente",
        color="success",
        dismissable=True,
        duration=3000
    )
    
    return (
        {'display': 'block'},
        gauge_fig,
        movie_info,
        trigger_cards,
        suitability_text,
        success_alert
    )


def run_server():
    """Start the dashboard server."""
    
    logger.info("Starting FilmLens Dashboard...")
    logger.info(f"Checking API connectivity...")
    
    if api_client.check_health():
        logger.info("API connection successful")
    else:
        logger.warning("Unable to connect to API - predictions may fail")
    
    logger.info(f"Dashboard running on http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    
    app.run_server(
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        debug=DEBUG_MODE
    )


if __name__ == "__main__":
    run_server()
