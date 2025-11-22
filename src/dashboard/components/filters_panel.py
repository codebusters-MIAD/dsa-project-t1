from dash import html, dcc
import dash_bootstrap_components as dbc


def create_filters_panel():
    """Create the filters panel with dropdowns for Genre, Year, Duration, Rating."""

    return html.Div(
        [
            html.H4(
                [
                    "Personaliza tu búsqueda ",
                    html.Span("🎨", style={"fontSize": "28px"}),
                ],
                className="text-center mb-4",
                style={"color": "#333", "fontWeight": "600"},
            ),
            dbc.Row(
                [
                    # Genero
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                id="filter-genre",
                                placeholder="Genero",
                                multi=True,
                                className="mb-3",
                                style={"borderRadius": "8px"},
                            )
                        ],
                        md=4,
                        className="mb-3",
                    ),
                    # Año
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                id="filter-year",
                                placeholder="Año",
                                className="mb-3",
                                style={"borderRadius": "8px"},
                            )
                        ],
                        md=4,
                        className="mb-3",
                    ),
                    # Duración
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                id="filter-duration",
                                placeholder="Duración",
                                className="mb-3",
                                style={"borderRadius": "8px"},
                            )
                        ],
                        md=4,
                        className="mb-3",
                    ),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    # Promedio Calificación
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                id="filter-rating",
                                placeholder="Promedio Calificación",
                                className="mb-3",
                                style={"borderRadius": "8px"},
                            )
                        ],
                        md=4,
                        className="mb-3",
                    ),
                    # Botón de búsqueda
                    dbc.Col(
                        [
                            dbc.Button(
                                "Buscar Películas",
                                id="btn-search-movies",
                                color="primary",
                                size="lg",
                                className="w-100",
                                style={"borderRadius": "8px", "fontWeight": "600"},
                            )
                        ],
                        md=8,
                        className="mb-3",
                    ),
                ],
            ),
            # Store para guardar las opciones de filtros
            dcc.Store(id="filter-options-store"),
        ],
        className="mb-5",
    )
