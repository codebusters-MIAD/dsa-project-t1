from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from config import TRIGGER_LABELS, TRIGGER_COLORS


def create_sensitivity_gauge(score: float, color: str):
    """Create a gauge chart for family suitability score."""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Indice de Aptitud Familiar", 'font': {'size': 20}},
        number={'suffix': "", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#fab1a0'},
                {'range': [40, 70], 'color': '#ffeaa7'},
                {'range': [70, 100], 'color': '#d5f4e6'}
            ],
            'threshold': {
                'line': {'color': "green", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="white",
        font={'color': "darkgray", 'family': "Arial"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_trigger_card(trigger_name: str, spanish_name: str, detected: bool, probability: float, color: str):
    """Create a card for individual trigger display."""
    
    status_text = "Tiene" if detected else "No tiene"
    status_color = "danger" if detected else "success"
    
    percentage = f"{probability * 100:.1f}%"
    
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.H5(spanish_name, className="card-title mb-2"),
                html.Div([
                    dbc.Badge(
                        status_text,
                        color=status_color,
                        className="me-2",
                        style={'fontSize': '14px', 'padding': '8px 12px'}
                    ),
                    html.Span(
                        percentage,
                        style={
                            'fontSize': '18px',
                            'fontWeight': 'bold',
                            'color': color
                        }
                    )
                ], className="d-flex align-items-center justify-content-between"),
                
                dbc.Progress(
                    value=probability * 100,
                    color="danger" if detected else "success",
                    className="mt-3",
                    style={'height': '8px'}
                )
            ])
        ])
    ], className="mb-3", style={'borderLeft': f'4px solid {color}'})


def create_results_display():
    """Create the results display section."""
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div(id="movie-info-display", className="mb-3"),
                    ])
                ], className="mb-4")
            ], md=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id="sensitivity-gauge")
                    ])
                ], className="mb-4")
            ], md=5),
            
            dbc.Col([
                html.H4("Que tan apta es esta pelicula para tu familia?", className="mb-4"),
                html.P(
                    id="family-suitability-text",
                    className="mb-4",
                    style={'fontSize': '16px', 'color': '#555'}
                ),
            ], md=7)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4("Lo que debes saber antes de verla", className="mb-4"),
                html.Div(id="triggers-display")
            ], md=12)
        ])
    ], id="results-container", style={'display': 'none'})
