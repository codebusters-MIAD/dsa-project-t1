from dash import html, dcc
import dash_bootstrap_components as dbc


def create_search_interface():
    """Create the main search interface with autocomplete."""

    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.Div(
                        [
                            html.I(className="bi bi-camera-reels", style={"fontSize": "48px", "marginRight": "15px"}),
                            html.H1("FilmLens", style={"display": "inline-block", "margin": "0"}),
                        ],
                        style={"display": "flex", "alignItems": "center", "justifyContent": "center"},
                    ),
                    html.P(
                        [
                            html.Span("🎯 ", style={"fontSize": "20px"}),
                            "Descubre películas que sí puedes ver en familia. En segundos.",
                        ],
                        className="lead text-center mt-2",
                        style={"color": "#666"},
                    ),
                ],
                className="text-center my-5",
            ),
            # Main search section
            html.Div(
                [
                    html.H3(
                        [
                            "Descubre qué hay detrás de cada película ",
                            html.Span("🎬", style={"fontSize": "32px"}),
                        ],
                        className="text-center mb-3",
                        style={"color": "#333", "fontWeight": "600"},
                    ),
                    # Description text
                    html.Div(
                        [
                            html.P(
                                [
                                    html.Strong("¿Buscas una película para ver en familia?"),
                                    " FilmLens te ayuda a ",
                                    html.Strong("filtrar lo que no quieres ver"),
                                    ", sin perder lo que sí te importa. Explora títulos, géneros y años, y ",
                                    html.Strong("déjate guiar por nuestro Índice de Aptitud Familiar"),
                                    ": una puntuación inteligente que te dice qué tan adecuada es cada película para niños entre 7 y 12 años.",
                                ],
                                className="text-center",
                                style={
                                    "color": "#555",
                                    "fontSize": "16px",
                                    "lineHeight": "1.6",
                                    "marginBottom": "10px",
                                },
                            ),
                            html.P(
                                [
                                    html.Strong("Sin spoilers. Sin complicaciones. Solo claridad."),
                                ],
                                className="text-center",
                                style={
                                    "color": "#333",
                                    "fontSize": "17px",
                                    "fontWeight": "600",
                                    "marginBottom": "20px",
                                },
                            ),
                        ],
                        className="mb-4",
                        style={"maxWidth": "800px", "margin": "0 auto"},
                    ),
                    # Search bar with autocomplete
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                "Seleccione el título que está buscando",
                                                style={
                                                    "fontSize": "13px",
                                                    "color": "#666",
                                                    "marginBottom": "8px",
                                                    "display": "block",
                                                },
                                            ),
                                            dcc.Input(
                                                id="search-input",
                                                type="text",
                                                placeholder="Ejemplo: Toy Story, The Dark Knight, Inception...",
                                                className="form-control form-control-lg",
                                                style={
                                                    "borderRadius": "8px",
                                                    "border": "2px solid #e0e0e0",
                                                    "fontSize": "16px",
                                                    "paddingLeft": "45px",
                                                },
                                            ),
                                            html.I(
                                                className="bi bi-search",
                                                style={
                                                    "position": "absolute",
                                                    "left": "15px",
                                                    "top": "50%",
                                                    "transform": "translateY(-50%)",
                                                    "fontSize": "20px",
                                                    "color": "#999",
                                                },
                                            ),
                                        ],
                                        style={"position": "relative"},
                                    ),
                                    # Autocomplete dropdown
                                    html.Div(
                                        id="autocomplete-dropdown",
                                        style={
                                            "display": "none",
                                            "position": "absolute",
                                            "top": "100%",
                                            "left": "0",
                                            "right": "0",
                                            "backgroundColor": "white",
                                            "border": "1px solid #ddd",
                                            "borderRadius": "0 0 8px 8px",
                                            "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
                                            "maxHeight": "300px",
                                            "overflowY": "auto",
                                            "zIndex": "1000",
                                            "marginTop": "5px",
                                        },
                                    ),
                                    # Hidden store for selected movie ID
                                    html.Div(id="selected-movie-id", style={"display": "none"}),
                                    # Predict button
                                    html.Div(
                                        dbc.Button(
                                            [
                                                html.I(className="bi bi-bar-chart-fill me-2"),
                                                "Predecir Sensibilidad Parental",
                                            ],
                                            id="btn-predict",
                                            color="primary",
                                            size="lg",
                                            className="w-100 mt-3",
                                            style={
                                                "borderRadius": "8px",
                                                "fontSize": "18px",
                                                "fontWeight": "600",
                                                "padding": "12px",
                                            },
                                            disabled=True,
                                        ),
                                        className="mt-3",
                                    ),
                                ],
                                style={"position": "relative"},
                            )
                        ],
                        className="mb-4",
                        style={"border": "none", "boxShadow": "0 2px 8px rgba(0,0,0,0.1)"},
                    ),
                ],
                className="mb-5",
            ),
        ]
    )


def create_autocomplete_item(movie: dict) -> dbc.ListGroupItem:
    """Create a single autocomplete suggestion item."""
    return dbc.ListGroupItem(
        [
            html.Strong(movie.get("movie_name", ""), style={"color": "#333"}),
            html.Span(
                f" ({movie.get('year', 'N/A')})",
                style={"color": "#999", "marginLeft": "5px", "fontSize": "14px"},
            ),
        ],
        id={"type": "autocomplete-item", "index": movie.get("imdb_id", "")},
        action=True,
        style={"cursor": "pointer", "border": "none", "borderBottom": "1px solid #f0f0f0"},
        className="autocomplete-item",
    )
