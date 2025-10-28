import os
import requests
import json
import time

# --- Configuración ---
# Guarda tu API key en una variable de entorno para mayor seguridad.
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "TU_API_KEY_AQUÍ")
BASE_URL = "https://api.themoviedb.org/3"
HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_API_KEY}"
}

def get_movie_ids_from_discover(pages=1):
    """Usa el endpoint /discover/movie para obtener IDs de películas."""
    movie_ids = []
    discover_url = f"{BASE_URL}/discover/movie"
    
    for page in range(1, pages + 1):
        params = {
            'language': 'es-ES',
            'sort_by': 'popularity.desc',
            'include_adult': False,
            'page': page
        }
        response = requests.get(discover_url, params=params, headers=HEADERS)
        response.raise_for_status()
        results = response.json().get('results', [])
        for movie in results:
            movie_ids.append(movie['id'])
        time.sleep(0.5) # Pequeña pausa para no saturar la API
        
    return movie_ids

def get_full_movie_details(movie_id):
    """Obtiene detalles y reseñas de una película por su ID."""
    movie_data = {}
    
    # 1. Obtener detalles principales
    details_url = f"{BASE_URL}/movie/{movie_id}"
    params = {'language': 'es-ES'}
    response = requests.get(details_url, params=params, headers=HEADERS)
    if response.status_code != 200:
        return None
    
    details = response.json()
    movie_data['tmdb_id'] = details.get('id')
    movie_data['imdb_id'] = details.get('imdb_id')
    movie_data['title'] = details.get('title')
    movie_data['overview'] = details.get('overview')
    movie_data['release_year'] = details.get('release_date', 'N/A')[:4]
    movie_data['genres'] = [genre['name'] for genre in details.get('genres', [])]

    # 2. Obtener reseñas
    reviews_url = f"{BASE_URL}/movie/{movie_id}/reviews"
    params = {}
    response = requests.get(reviews_url, params=params, headers=HEADERS)
    if response.status_code == 200:
        reviews = response.json().get('results', [])
        movie_data['reviews'] = [
            {'author': r['author'], 'content': r['content']} for r in reviews
        ]
    else:
        movie_data['reviews'] = []
        
    return movie_data

if __name__ == "__main__":
    if TMDB_API_KEY == "TU_API_KEY_AQUÍ":
        print("Error: Por favor, configura tu clave de API de TMDB.")
    else:
        print("Iniciando la recolección de datos...")
        # Obtenemos IDs de las películas de la primera página de "discover"
        movie_ids_to_fetch = get_movie_ids_from_discover(pages=1)
        
        all_movies_data = []
        
        print(f"Se encontraron {len(movie_ids_to_fetch)} películas. Obteniendo detalles...")
        
        for i, movie_id in enumerate(movie_ids_to_fetch):
            print(f"Procesando película {i+1}/{len(movie_ids_to_fetch)} (ID: {movie_id})...")
            full_details = get_full_movie_details(movie_id)
            if full_details:
                all_movies_data.append(full_details)
            time.sleep(0.5) # Respetar los límites de la API

        # Guardar los datos en un archivo JSON en la carpeta data/raw
        output_path = os.path.join('data', 'raw', 'tmdb_data.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_movies_data, f, ensure_ascii=False, indent=4)
            
        print(f"\nProceso completado. Datos guardados en: {output_path}")

