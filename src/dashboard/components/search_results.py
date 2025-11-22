from dash import html
import dash_bootstrap_components as dbc


def create_search_results():
    """Create the search results section."""

    return html.Div(
        [
            # Loading indicator
            dbc.Spinner(
                html.Div(id="search-results-content"),
                color="primary",
                type="border",
                fullscreen=False,
            ),
        ],
        id="search-results-container",
        style={"display": "none"},
        className="mt-5",
    )


def create_movie_card(movie: dict) -> dbc.Card:
    """Create a movie card for search results."""

    genres = ", ".join(movie.get("genre", [])[:3]) if movie.get("genre") else "N/A"
    directors = ", ".join(movie.get("director", [])[:2]) if movie.get("director") else "N/A"
    rating = movie.get("rating", 0)
    year = movie.get("year", "N/A")
    runtime = movie.get("runtime", 0)

    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.H5(movie.get("movie_name", "Unknown"), className="card-title mb-2"),
                            html.Div(
                                [
                                    dbc.Badge(
                                        f"⭐ {rating:.1f}" if rating else "N/A",
                                        color="warning",
                                        className="me-2",
                                    ),
                                    dbc.Badge(f"📅 {year}", color="info", className="me-2"),
                                    dbc.Badge(f"⏱️ {runtime} min" if runtime else "N/A", color="secondary"),
                                ],
                                className="mb-3",
                            ),
                            html.P(
                                [html.Strong("Género: "), html.Span(genres)],
                                className="mb-2",
                                style={"fontSize": "14px"},
                            ),
                            html.P(
                                [html.Strong("Director: "), html.Span(directors)],
                                className="mb-3",
                                style={"fontSize": "14px"},
                            ),
                            html.P(
                                movie.get("description", "Sin descripción disponible")[:200] + "...",
                                className="text-muted mb-3",
                                style={"fontSize": "13px"},
                            ),
                            dbc.Button(
                                "Analizar Contenido",
                                id={"type": "analyze-movie-btn", "index": movie.get("imdb_id", "")},
                                color="primary",
                                size="sm",
                                className="w-100",
                            ),
                        ]
                    )
                ]
            )
        ],
        className="mb-3",
        style={"boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "transition": "all 0.3s"},
    )
