from dash import html, dcc
import dash_bootstrap_components as dbc


def create_input_form():
    """Create the movie information input form."""
    
    return dbc.Card([
        dbc.CardBody([
            html.H4("Informacion de la Pelicula", className="card-title mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("ID de Pelicula", className="form-label"),
                    dbc.Input(
                        id="input-movie-id",
                        type="text",
                        placeholder="Ej: tt0468569",
                        className="mb-3"
                    ),
                ], md=6),
                
                dbc.Col([
                    dbc.Label("Genero", className="form-label"),
                    dcc.Dropdown(
                        id="input-genre",
                        options=[
                            {'label': 'Accion', 'value': 'Action'},
                            {'label': 'Aventura', 'value': 'Adventure'},
                            {'label': 'Animacion', 'value': 'Animation'},
                            {'label': 'Biografia', 'value': 'Biography'},
                            {'label': 'Comedia', 'value': 'Comedy'},
                            {'label': 'Crimen', 'value': 'Crime'},
                            {'label': 'Documental', 'value': 'Documentary'},
                            {'label': 'Drama', 'value': 'Drama'},
                            {'label': 'Familia', 'value': 'Family'},
                            {'label': 'Fantasia', 'value': 'Fantasy'},
                            {'label': 'Historia', 'value': 'History'},
                            {'label': 'Terror', 'value': 'Horror'},
                            {'label': 'Musical', 'value': 'Musical'},
                            {'label': 'Misterio', 'value': 'Mystery'},
                            {'label': 'Romance', 'value': 'Romance'},
                            {'label': 'Ciencia Ficcion', 'value': 'Sci-Fi'},
                            {'label': 'Deportes', 'value': 'Sport'},
                            {'label': 'Thriller', 'value': 'Thriller'},
                            {'label': 'Guerra', 'value': 'War'},
                            {'label': 'Western', 'value': 'Western'},
                        ],
                        placeholder="Seleccione un genero",
                        className="mb-3"
                    ),
                ], md=6),
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Titulo de la Pelicula", className="form-label"),
                    dbc.Input(
                        id="input-title",
                        type="text",
                        placeholder="Ej: The Dark Knight",
                        className="mb-3"
                    ),
                ], md=12),
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Descripcion de la Pelicula", className="form-label"),
                    dbc.Textarea(
                        id="input-description",
                        placeholder="Ingrese la descripcion de la pelicula (minimo 10 caracteres)...",
                        style={'height': '120px'},
                        className="mb-3"
                    ),
                ], md=12),
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Analizar Pelicula",
                        id="btn-analyze",
                        color="primary",
                        size="lg",
                        className="w-100"
                    ),
                ], md=12),
            ]),
        ])
    ], className="mb-4")
