from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


def create_sensitivity_gauge(score: float, color: str):
    """Create a gauge chart for family suitability score."""

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Indice de Aptitud Familiar", "font": {"size": 20}},
            number={"suffix": "", "font": {"size": 40}},
            gauge={
                "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "darkgray"},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 40], "color": "#fab1a0"},
                    {"range": [40, 70], "color": "#ffeaa7"},
                    {"range": [70, 100], "color": "#d5f4e6"},
                ],
                "threshold": {"line": {"color": "green", "width": 4}, "thickness": 0.75, "value": 70},
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="white",
        font={"color": "darkgray", "family": "Arial"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def create_trigger_card_multilevel(trigger_name: str, spanish_name: str, icon: str, nivel: str, probability: float, probabilities_all: dict, description: str):
    """Create a card for individual trigger display with multilevel support."""
    
    # Map level to badge text and Bootstrap color
    level_config = {
        "sin_contenido": {
            "text": "Sin Contenido",
            "color": "success"  # Verde
        },
        "moderado": {
            "text": "Moderado",
            "color": "warning"  # Naranja
        },
        "alto": {
            "text": "Alto",
            "color": "danger"  # Rojo
        },
    }
    
    badge_text = level_config[nivel]["text"]
    badge_color = level_config[nivel]["color"]
    
    return html.Div(
        [
            # Header with icon
            html.Div(
                [
                    html.Span(icon, style={"fontSize": "40px"}),
                ],
                className="text-center mb-2",
            ),
            # Title
            html.H6(
                spanish_name,
                className="text-center mb-3",
                style={"fontWeight": "600", "fontSize": "16px"}
            ),
            # Description
            html.P(
                description,
                className="text-center mb-3",
                style={
                    "fontSize": "12px",
                    "lineHeight": "1.4",
                    "color": "#666",
                    "minHeight": "55px"
                },
            ),
            # Badge usando color de Bootstrap
            html.Div(
                [
                    dbc.Badge(
                        badge_text,
                        color=badge_color,
                        className="px-3 py-2",
                        style={
                            "fontSize": "14px",
                            "fontWeight": "600",
                            "borderRadius": "20px",
                        },
                    )
                ],
                className="text-center",
            ),
        ],
        style={
            "padding": "20px 15px",
            "borderRadius": "8px",
            "border": "1px solid #e0e0e0",
            "backgroundColor": "white",
            "height": "100%",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
        },
    )


def create_results_display():
    """Create the results display section."""

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.Div(id="movie-info-display", className="mb-3"),
                                        ]
                                    )
                                ],
                                className="mb-4",
                            )
                        ],
                        md=12,
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col([dbc.Card([dbc.CardBody([dcc.Graph(id="sensitivity-gauge")])], className="mb-4")], md=5),
                    dbc.Col(
                        [
                            html.H4("Que tan apta es esta pelicula para tu familia?", className="mb-4"),
                            html.P(
                                id="family-suitability-text",
                                className="mb-4",
                                style={"fontSize": "16px", "color": "#555"},
                            ),
                        ],
                        md=7,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Lo que debes saber antes de verla", className="mb-4"),
                            html.Div(id="triggers-display"),
                        ],
                        md=12,
                    )
                ]
            ),
        ],
        id="results-container",
        style={"display": "none"},
    )
