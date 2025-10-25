"""
ml_ddd_dataset_builder.py
===========================

This module contains utilities for downloading and preparing the ML-DDD
dataset of movies with DoesTheDogDie (DDD) content warnings. In addition to generating
the sensitivity and interaction tables, this script demonstrates how to
enrich the dataset with basic MovieLens metadata (titles and genres),
map the many DDD triggers into broader categories (e.g. violence,
drogas, suicidio) and compute a simple "aptitud familiar" score.  The
code is designed as a standalone module that you can run directly or
import into a larger pipeline.

Usage
-----

Running this module as a script will download the required resources
into a local ``ml_ddd_data`` directory, generate the sensitivity and
interaction tables, add summary statistics, join movie titles and
genres from MovieLens, derive the broader content categories and
aptitud scores, and finally export a single CSV file called
``peliculas_sensibilidad_gold.csv`` containing the enriched dataset.

Important notes
---------------

* This script requires internet access to download the MovieLens 25M
  dataset and the pickled DDD dictionary from HuggingFace.
* Requires the ``huggingface_hub`` library to use ``hf_hub_download``.
  A HuggingFace token may be required via environment variable or login
  if the dataset is gated.
* The mapping from the 137 DDD warnings to high-level categories is
  provided as a starting point and should be refined for the use case.
* Fetching full plot summaries is not implemented here. The dataset can be
  augmented with synopses using APIs like TMDb or OMDb when API keys are
  available. Placeholder code is included for guidance.
"""
x
from __future__ import annotations

import os
import pathlib
import pickle
import shutil
import tempfile
from typing import Dict, Iterable, List, Tuple
from zipfile import ZipFile

import pandas as pd
import requests
from huggingface_hub import hf_hub_download
import urllib.request
import time

# Directory to store intermediate and final data
INPUT_DATA_DIR = pathlib.Path("../data/raw/ml").resolve()

# URL for MovieLens 25M dataset
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


def download_movielens_with_retries(url: str, zip_path: pathlib.Path, max_retries: int = 3, verbose: bool = True) -> None:
    """Download with retries using urllib for more stability"""
    for attempt in range(max_retries):
        try:
            if verbose:
                print(f"Descarga intento {attempt + 1}/{max_retries}...")
            
            def progress_hook(block_num, block_size, total_size):
                if verbose and total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(100.0, 100 * downloaded / total_size)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"  Downloaded {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='\r')
            
            urllib.request.urlretrieve(url, zip_path, progress_hook)
            if verbose:
                print(f"\n✅ Download completed successfully")
            return
            
        except Exception as e:
            if verbose:
                print(f"\n❌ Intento {attempt + 1} falló: {e}")
            if zip_path.exists():
                zip_path.unlink()
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to download after {max_retries} attempts: {e}")
            time.sleep(5)  # Esperar antes del siguiente intento

def download_movielens(mkdir: bool = True, verbose: bool = False) -> None:
    """Download and unzip the MovieLens 25M dataset into ``INPUT_DATA_DIR``.

    If ``mkdir`` is True, the output directory will be created if it
    doesn't exist.  Setting ``verbose`` prints incremental progress.
    """
    output_dir = INPUT_DATA_DIR
    if not output_dir.exists():
        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise RuntimeError(f"{output_dir} does not exist. Pass mkdir=True or create it yourself.")

    # # Descarga comentada - usando archivos locales directamente
    # url = MOVIELENS_URL
    # zip_path = output_dir / "ml-25m.zip"
    
    # if verbose:
    #     print(f"Downloading MovieLens 25M from {url}")
    #     print("Usando método robusto con reintentos...")
    
    # try:
    #     # Usar descarga con reintentos
    #     download_movielens_with_retries(url, zip_path, max_retries=3, verbose=verbose)
        
    #     # Extraer el archivo ZIP
    #     if verbose:
    #         print("Extracting MovieLens archive...")
        
    #     with ZipFile(zip_path, 'r') as zipf:
    #         zipf.extractall(output_dir)
        
    #     # Limpiar archivo ZIP después de extraer
    #     zip_path.unlink()
        
    #     if verbose:
    #         print(f"✅ MovieLens extracted to {output_dir}")
            
    # except Exception as e:
    #     if zip_path.exists():
    #         zip_path.unlink()  # Limpiar archivo parcial
    #     raise RuntimeError(f"Failed to download MovieLens dataset: {e}")
    
    if verbose:
        print("Saltando descarga - usando archivos CSV existentes...")


def download_ml_ratings() -> pd.DataFrame:
    """Download MovieLens and return the ratings DataFrame.

    Returns
    -------
    DataFrame
        Pandas DataFrame with columns (userId, movieId, rating, timestamp).
    """
    ratings_path = INPUT_DATA_DIR / "ml-25m" / "ratings.csv"
    
    # COMENTADO: Funciones de descarga (archivos ya existen)
    # if not ratings_path.exists():
    #     print("DOWNLOADING MOVIELENS 25M...")
    #     download_movielens(verbose=True)
    # else:
    #     print("MovieLens 25M ya existe, usando archivo local...")
    
    print("Cargando ratings desde archivo local...")
    ratings_df = pd.read_csv(ratings_path)
    print("COMPLETED LOAD")
    return ratings_df


def download_ddd_warnings() -> Dict[str, Dict[int, Dict[str, int]]]:
    """Download the DoesTheDogDie warning dictionary from HuggingFace.

    Returns
    -------
    dict
        Dictionary mapping warning names to a nested dict of work IDs and
        vote counts.  Each entry ``votes`` contains keys 'yesSum' and
        'noSum'.
    """
    dest_path = INPUT_DATA_DIR / "ddd_dict.pkl"
    
    if not dest_path.exists():
        print("DOWNLOADING DOES THE DOG DIE WARNING DICTIONARY...")
        repo_id = "sdeangroup/NavigatingSensitivity"
        filename = "ddd_dict.pkl"
        try:
            # hf_hub_download returns a local path to the file
            ddd_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
            # Copy into our data directory for convenience
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ddd_path, dest_path)
            print(f"✅ DDD dictionary descargado exitosamente a: {dest_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download DDD dictionary: {e}")
    else:
        print("DDD dictionary ya existe, usando archivo local...")
    
    print("Cargando DDD dictionary...")
    with open(dest_path, 'rb') as handle:
        ddd_dict = pickle.load(handle)
    
    print("COMPLETED DOWNLOAD")
    return ddd_dict


def get_warning_votes(votes: Dict[str, int]) -> Tuple[int, int, int, int]:
    """Convert yes/no vote counts into one‑hot categories.

    The DDD dictionary stores counts of 'yes' and 'no' votes for each
    trigger.  We convert these into four binary indicators: clear yes,
    clear no, unclear, and no votes.  A clear consensus requires that
    one side has >75% of the total votes.  No votes returns the last
    indicator.
    """
    total = votes['yesSum'] + votes['noSum']
    majority_threshold = 0.75 * total
    if total == 0:
        return (0, 0, 0, 1)
    if votes['yesSum'] > majority_threshold:
        return (1, 0, 0, 0)
    if votes['noSum'] > majority_threshold:
        return (0, 1, 0, 0)
    return (0, 0, 1, 0)


def get_sensitivity_table(ddd_dict: Dict[str, Dict[int, Dict[str, int]]]) -> pd.DataFrame:
    """Build the sensitivity table DataFrame from the DDD dictionary.

    The resulting DataFrame has one row per ``work_id`` (MovieLens
    ``movieId``) and four columns per warning (``Clear Yes: <warning>``,
    ``Clear No: <warning>``, ``Unclear: <warning>``, ``No Votes:
    <warning>``), encoded as 0/1 integers.
    """
    data: Dict[int, Dict[str, int]] = {}
    # Iterate over all warnings and their associated works
    for warning, work_votes in ddd_dict.items():
        for work_id, votes in work_votes.items():
            if work_id not in data:
                data[work_id] = {}
            c_yes, c_no, unclear, no_votes = get_warning_votes(votes)
            data[work_id][f"Clear Yes: {warning}"] = c_yes
            data[work_id][f"Clear No: {warning}"] = c_no
            data[work_id][f"Unclear: {warning}"] = unclear
            data[work_id][f"No Votes: {warning}"] = no_votes
    sensitivity_table = pd.DataFrame(data).T.fillna(0).astype(int)
    sensitivity_table.reset_index(inplace=True)
    sensitivity_table.rename(columns={'index': 'work_id'}, inplace=True)
    return sensitivity_table


def filter_tables(
    sensitivity_table: pd.DataFrame, interaction_table: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter to users and works with at least three interactions.

    The MovieLens ratings table is renamed to use ``user_id`` and
    ``work_id`` columns.  Iteratively removes users with fewer than
    three ratings and then removes works that appear in fewer than
    three ratings.  Returns the filtered versions of both tables.
    """
    interaction_table = interaction_table.rename(columns={"userId": "user_id", "movieId": "work_id"})
    print(f"Initial number of users before filtering: {interaction_table['user_id'].nunique()}")
    print(f"Initial number of works before filtering: {len(sensitivity_table)}")
    for i in range(3):
        # Keep users with ≥3 interactions
        user_counts = interaction_table['user_id'].value_counts()
        users_to_keep = user_counts[user_counts >= 3].index
        interaction_table = interaction_table[interaction_table['user_id'].isin(users_to_keep)]
        print(f"Number of users with ≥3 interactions (pass {i+1}): {interaction_table['user_id'].nunique()}")
        # Keep works with ≥3 interactions
        work_counts = interaction_table['work_id'].value_counts()
        works_to_keep = work_counts[work_counts >= 3].index
        sensitivity_table = sensitivity_table[sensitivity_table['work_id'].isin(works_to_keep)]
        print(f"Number of works with ≥3 interactions (pass {i+1}): {len(sensitivity_table)}")
        # Restrict interactions to remaining works
        interaction_table = interaction_table[interaction_table['work_id'].isin(works_to_keep)]
    print(f"Final number of users: {interaction_table['user_id'].nunique()}")
    print(f"Final number of works: {len(sensitivity_table)}")
    print(f"Final number of interactions: {len(interaction_table)}")
    return sensitivity_table, interaction_table


def add_summary_stats(
    sensitivity_table: pd.DataFrame, interaction_table: pd.DataFrame
) -> pd.DataFrame:
    """Add rating count and average rating to the sensitivity table.

    Also adds a ``user_ratings`` column containing a dictionary mapping
    user IDs to their rating for each work.  If a work has no
    remaining ratings, the dictionary is empty.
    """
    work_ratings_summary = interaction_table.groupby('work_id').agg(
        n_ratings=('rating', 'count'),
        av_rating=('rating', 'mean'),
    ).reset_index()
    sensitivity_table = pd.merge(sensitivity_table, work_ratings_summary, on='work_id', how='left')
    
    # OPTIMIZACIÓN: Agregar columna vacía para user_ratings para evitar problemas de memoria
    # En datasets grandes, este diccionario puede consumir demasiada RAM
    sensitivity_table['user_ratings'] = [{}] * len(sensitivity_table)
    
    # Llenar NaN en caso de works sin ratings después del merge
    sensitivity_table['n_ratings'] = sensitivity_table['n_ratings'].fillna(0)
    sensitivity_table['av_rating'] = sensitivity_table['av_rating'].fillna(0.0)
    
    return sensitivity_table


def load_movies_metadata() -> pd.DataFrame:
    """Load MovieLens movie metadata (title and genres).

    Returns
    -------
    DataFrame
        Contains columns ``work_id``, ``title``, and ``genres``.  The
        ``work_id`` column matches the MovieLens ``movieId``.
    """
    movies_path = INPUT_DATA_DIR / "ml-25m" / "movies.csv"
    movies_df = pd.read_csv(movies_path)
    movies_df.rename(columns={'movieId': 'work_id'}, inplace=True)
    return movies_df[['work_id', 'title', 'genres']]


def load_links_metadata() -> pd.DataFrame:
    """Load MovieLens links metadata (IMDb and TMDb IDs).

    This table can be used to look up external metadata such as plot
    summaries via external APIs (e.g. TMDb or OMDb).  The actual
    fetching of synopses is left as an exercise and not implemented
    here.
    """
    links_path = INPUT_DATA_DIR / "ml-25m" / "links.csv"
    links_df = pd.read_csv(links_path)
    links_df.rename(columns={'movieId': 'work_id'}, inplace=True)
    return links_df[['work_id', 'imdbId', 'tmdbId']]


def map_warnings_to_categories(
    sensitivity_table: pd.DataFrame,
    category_map: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Agrupa múltiples advertencias DDD en categorías de alto nivel.
    Para cada categoría agrega una columna binaria cat_<categoria> = 1
    si alguna advertencia mapeada aparece marcada "Clear Yes".
    """
    # 1) Identificar columnas "Clear Yes"
    clear_yes_cols = [c for c in sensitivity_table.columns if c.startswith("Clear Yes: ")]

    # 2) Precomputar, por fila, la LISTA de nombres de advertencia (en minúsculas) con Clear Yes = 1
    def clear_yes_names_lower(row) -> list:
        names = []
        for c in clear_yes_cols:
            val = row[c]
            # robustez: acepta 1 o "1"
            if val == 1 or (isinstance(val, str) and val.strip() == "1"):
                # nombre completo sin el prefijo
                name = c.replace("Clear Yes: ", "")
                names.append(name.lower())
        return names

    yes_names_col = sensitivity_table.apply(clear_yes_names_lower, axis=1)

    # 3) Normalizar triggers a minúsculas y usar SUBCADENA
    category_map_lc = {
        cat: [t.lower() for t in triggers] for cat, triggers in category_map.items()
    }

    for category, triggers in category_map_lc.items():
        col_name = f"cat_{category}"
        sensitivity_table[col_name] = yes_names_col.apply(
            lambda names: int(any(
                any(trigger in name for trigger in triggers)
                for name in names
            ))
        )

    return sensitivity_table


def calculate_aptitude(
    row: pd.Series, weights: Dict[str, float], severity_cols: Iterable[str] = None
) -> float:
    """Compute a simple 0–100 aptitud familiar score.

    ``weights`` assigns a weight to each category (should sum to ≤1).
    If ``severity_cols`` is provided, it should map to columns of
    severity on a 0–1 scale; otherwise binary category columns are
    used.  The score is 100 * (1 - weighted sum of severities).
    """
    risk = 0.0
    for cat, weight in weights.items():
        # Prefer severity columns if provided; else use binary indicator
        sev_col = severity_cols[cat] if severity_cols else None
        sev_val = row.get(sev_col, row.get(f"cat_{cat}", 0))
        risk += weight * float(sev_val)
    # Ensure score stays in [0, 100]
    score = max(0.0, 100.0 * (1.0 - risk))
    return round(score, 2)


def compute_category_levels(
    df: pd.DataFrame,
    category_map: Dict[str, List[str]],
    unclear_weight: float = 0.5,
    thresholds: Tuple[float, float] = (0.34, 0.67),
    level_suffix: str = "_nivel",
    score_suffix: str = "_score",
) -> pd.DataFrame:
    """
    Calcula <categoria>_score en [0,1] y <categoria>_nivel ('leve','medio','alto')
    para cada categoría definida en category_map. Normaliza las cadenas para
    asegurarse de que las advertencias de DDD coincidan con los patrones.
    """

    import re

    # Función de normalización: elimina todo excepto letras y números
    def normalize_string(s: str) -> str:
        return re.sub(r'[^a-z0-9]', '', s.lower())

    # Columnas base existentes (todas las advertencias)
    clear_yes_cols = [c for c in df.columns if c.startswith("Clear Yes: ")]
    clear_no_cols  = [c for c in df.columns if c.startswith("Clear No: ")]
    unclear_cols   = [c for c in df.columns if c.startswith("Unclear: ")]

    # Catálogo de nombres de advertencias (sin prefijo) y su versión normalizada
    triggers_present = [c[len("Clear Yes: "):] for c in clear_yes_cols]
    triggers_norm_map = {trg: normalize_string(trg) for trg in triggers_present}

    # Normalizar los patrones de category_map
    category_map_norm = {
        cat: [normalize_string(t) for t in triggers]
        for cat, triggers in category_map.items()
    }

    # Conversor numérico para DataFrames/Series
    def to_num(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        elif isinstance(obj, pd.Series):
            return pd.to_numeric(obj, errors="coerce").fillna(0.0)
        else:
            return obj

    t1, t2 = thresholds

    for category, substrings_norm in category_map_norm.items():
        # 1) Encontrar advertencias que pertenezcan a la categoría (comparación normalizada)
        matched_triggers = [
            trg for trg, norm in triggers_norm_map.items()
            if any(sub in norm for sub in substrings_norm)
        ]

        # 2) Si no hay advertencias emparejadas, score=0 y nivel=leve
        if not matched_triggers:
            df[f"{category}{score_suffix}"] = 0.0
            df[f"{category}{level_suffix}"] = "leve"
            continue

        # 3) Agrupar columnas Clear Yes, Clear No y Unclear para esas advertencias
        yes_cols = [f"Clear Yes: {trg}" for trg in matched_triggers if f"Clear Yes: {trg}" in df.columns]
        no_cols  = [f"Clear No: {trg}"  for trg in matched_triggers if f"Clear No: {trg}"  in df.columns]
        unc_cols = [f"Unclear: {trg}"   for trg in matched_triggers if f"Unclear: {trg}"   in df.columns]

        Y = to_num(df[yes_cols]) if yes_cols else 0.0
        N = to_num(df[no_cols])  if no_cols  else 0.0
        U = to_num(df[unc_cols]) if unc_cols else 0.0

        # 4) Denominador = cantidad de advertencias con alguna votación (Yes, No, Unclear)
        if isinstance(Y, pd.DataFrame) and len(yes_cols) > 0:
            # Calcular sumas de manera más robusta
            y_sum = Y.sum(axis=1)
            n_sum = N.sum(axis=1)
            u_sum = U.sum(axis=1)
            denom = y_sum + n_sum + u_sum
            numer = y_sum + unclear_weight * u_sum
        else:
            denom = pd.Series([0.0] * len(df), index=df.index)
            numer = pd.Series([0.0] * len(df), index=df.index)

        score = (numer / denom.replace(0, pd.NA)).fillna(0.0).clip(0, 1)
        nivel = pd.cut(
            score,
            bins=[-1, t1, t2, float('inf')],
            labels=["leve", "medio", "alto"],
            right=False
        )

        df[f"{category}{score_suffix}"] = score.round(3)
        df[f"{category}{level_suffix}"] = nivel.astype(str)

    return df


def build_dataset() -> pd.DataFrame:
    """High‑level orchestration of the dataset building pipeline.

    Downloads the raw data, constructs the sensitivity and interaction
    tables, filters out sparse users and works, adds summary
    statistics, enriches with MovieLens metadata, maps warnings to
    categories, computes the aptitud score and returns the final
    DataFrame.
    """
    # Step 1: download raw data
    ratings_df = download_ml_ratings()
    ddd_dict = download_ddd_warnings()
    # Step 2: build sensitivity table
    sensitivity_table = get_sensitivity_table(ddd_dict)
    # Step 3: filter interactions and sensitivity table
    sensitivity_table, interaction_table = filter_tables(sensitivity_table, ratings_df)
    # Step 4: add summary stats
    sensitivity_table = add_summary_stats(sensitivity_table, interaction_table)
    # Step 5: join MovieLens metadata (title, genres, IMDb/TMDb IDs)
    movies_meta = load_movies_metadata()
    links_meta = load_links_metadata()
    merged = sensitivity_table.merge(movies_meta, on='work_id', how='left')
    merged = merged.merge(links_meta, on='work_id', how='left')
    # Step 6: derive category columns
    # Define your mapping from the 137 DDD warnings to broader categories
    category_map = {
        "violencia": [
            "blood/gore", "gun violence", "domestic violence", "torture",
            "decapitation", "excessive gore", "heads get squashed",
            "violence", "fight", "shoot", "stab", "kill",
        ],
        "drogas": [
            "abuse alcohol", "use drugs", "addiction", "alcohol",
            "drug", "overdose", "syringes", "needle",
        ],
        "suicidio": [
            "die by suicide", "attempt suicide", "kill myself",
        ],
        "lenguaje_fuerte": [
            "obscene language", "n-word", "slurs", "transphobic", "homophobic",
            "ableist language",
        ],
        "sexualidad": [
            "sexual content", "sexually assaulted", "raped", "nude scenes",
            "incest", "sexualized", "pornography", "sexual abuse",
        ],
    }
    merged = map_warnings_to_categories(merged, category_map)
    
    # Calcular columnas de score y nivel por categoría
    merged = compute_category_levels(
        merged,
        category_map=category_map,
        unclear_weight=0.5,
        thresholds=(0.34, 0.67),
        level_suffix="_nivel",
        score_suffix="_score",
    )
    # Step 7: compute aptitud familiar (weights sum to 1)
    weights = {
        "violencia": 0.30,
        "drogas": 0.20,
        "suicidio": 0.25,
        "lenguaje_fuerte": 0.15,
        "sexualidad": 0.10,
    }
    merged['aptitud_familiar'] = merged.apply(lambda r: calculate_aptitude(r, weights), axis=1)
    return merged


def main() -> None:
    """Entry point for running the script from the command line."""
    try:
        if not INPUT_DATA_DIR.exists():
            INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        print("Iniciando construcción del dataset...")
        final_df = build_dataset()
        
        # Crear directorio processed/ml y guardar el dataset final allí
        processed_dir = INPUT_DATA_DIR.parent.parent / "processed" / "ml"
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = processed_dir / "peliculas_sensibilidad_gold.csv"
        final_df.to_csv(out_path, index=False)
        print(f"Dataset guardado exitosamente en: {out_path}")
        print(f"Dimensiones finales: {final_df.shape}")
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        raise


if __name__ == "__main__":
    main()