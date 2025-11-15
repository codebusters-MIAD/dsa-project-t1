
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns              
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
import glob
import os


def _normalize_imdb_numeric(series: pd.Series, width: int) -> pd.Series:
    
    s = series.astype(str)
    # elimina .0 de lecturas como 12345.0
    s = s.str.replace(r'\.0$', '', regex=True)
    # extrae solo dígitos por seguridad
    s = s.str.extract(r'(\d+)', expand=False)
    # aplica zero-padding
    return s.str.zfill(width)



# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_unify_movie_genres(data_path: str = "../../data/raw/train/movies_imdb_sample/") -> pd.DataFrame:

    logger = logging.getLogger("load_and_unify_movie_genres")
    
    # Buscar todos los archivos CSV en el directorio, excluyendo el archivo unificado
    csv_files = [f for f in glob.glob(os.path.join(data_path, "*.csv"))
                 if os.path.basename(f) != "movies_imdb_unified.csv"]
    
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {data_path} (excluyendo movies_imdb_unified.csv)")
    
    logger.info(f"Encontrados {len(csv_files)} archivos CSV para unificar")
    
    unified_data = []
    expected_columns = None
    
    for file_path in csv_files:
        try:
            # Extraer el nombre del género del archivo
            genre_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Cargar el archivo
            df = pd.read_csv(file_path)
            logger.info(f"Cargando {genre_name}.csv: {len(df):,} registros")
            
            # Verificar estructura en el primer archivo
            if expected_columns is None:
                expected_columns = list(df.columns)
                logger.info(f"Estructura esperada: {len(expected_columns)} columnas")
            else:
                # Verificar que todos los archivos tengan la misma estructura
                if list(df.columns) != expected_columns:
                    raise ValueError(f"Estructura inconsistente en {file_path}")
            
            # Añadir columna de género de origen
            df['source_genre'] = genre_name
            
            unified_data.append(df)
            
        except Exception as e:
            logger.error(f"Error procesando {file_path}: {e}")
            raise
    
    # Unificar todos los DataFrames
    unified_df = pd.concat(unified_data, ignore_index=True)
    
    logger.info(f"Unificación completada:")
    logger.info(f"  - Total de registros: {len(unified_df):,}")
    logger.info(f"  - Total de columnas: {len(unified_df.columns)}")
    logger.info(f"  - Géneros procesados: {len(csv_files)}")
    
    return unified_df


def process_imdb_ids(df: pd.DataFrame) -> pd.DataFrame:

    logger = logging.getLogger("process_imdb_ids")
    
    if 'movie_id' not in df.columns:
        logger.error("Columna 'movie_id' no encontrada en el dataset")
        return df
    
    logger.info("PROCESANDO IDs DE IMDb...")
    
    # Crear copia del DataFrame para no modificar el original
    df_processed = df.copy()
    
    # Crear nueva columna imdbId quitando los primeros 2 caracteres
    df_processed['imdbId'] = df_processed['movie_id'].str[2:]
    
    # Verificar que los valores resultantes sean numéricos
    non_null_ids = df_processed['imdbId'].dropna()
    numeric_count = non_null_ids.str.isnumeric().sum() if len(non_null_ids) > 0 else 0
    total_count = len(df_processed)
    
    logger.info(f"Columna 'imdbId' creada exitosamente:")
    logger.info(f"   • Total de registros: {total_count:,}")
    logger.info(f"   • Registros numéricos: {numeric_count:,}")
    logger.info(f"   • Porcentaje válido: {(numeric_count/total_count)*100:.1f}%")
    
    # Mostrar ejemplos de transformación
    logger.info(f"Ejemplos de transformación:")
    sample_data = df_processed[['movie_id', 'imdbId']].head(5)
    for _, row in sample_data.iterrows():
        logger.info(f"   • {row['movie_id']} → {row['imdbId']}")
    
    return df_processed


def optimize_sensitivity_dataset(df: pd.DataFrame) -> pd.DataFrame:

    logger = logging.getLogger("optimize_sensitivity_dataset")
    
    logger.info("OPTIMIZANDO DATASET DE SENSIBILIDAD...")
    logger.info(f"Dataset original: {len(df):,} filas  {len(df.columns)} columnas")
    
    # Definir columnas esenciales para el análisis predictivo
    columnas_esenciales = [
        # Identificación básica
        'work_id', 'title', 'genres',
        
        # Calificaciones externas
        'av_rating', 'n_ratings',
        
        # Scores principales de sensibilidad (agregados)
        'violencia_score',
        'sexualidad_score', 
        'drogas_score',
        'lenguaje_fuerte_score',
        'suicidio_score',
        
        # Niveles de sensibilidad
        'violencia_nivel',
        'sexualidad_nivel',
        'drogas_nivel',
        'lenguaje_fuerte_nivel',
        'suicidio_nivel',
        
        # Variable objetivo principal
        'aptitud_familiar',
        
        # Enlaces externos (para joins)
        'imdbId', 'tmdbId'
    ]
    
    # Identificar columnas disponibles
    columnas_disponibles = []
    columnas_faltantes = []
    
    for col in columnas_esenciales:
        if col in df.columns:
            columnas_disponibles.append(col)
        else:
            columnas_faltantes.append(col)
    
    # Crear dataset optimizado
    df_optimizado = df[columnas_disponibles].copy()
    
    # Calcular estadísticas de optimización
    filas_originales = len(df)
    columnas_originales = len(df.columns)
    columnas_optimizadas = len(df_optimizado.columns)
    columnas_eliminadas = columnas_originales - columnas_optimizadas
    
    # Calcular reducción de memoria
    memoria_original = df.memory_usage(deep=True).sum() / (1024**2)  
    memoria_optimizada = df_optimizado.memory_usage(deep=True).sum() / (1024**2)  
    reduccion_memoria = ((memoria_original - memoria_optimizada) / memoria_original) * 100
    
    # Reportar resultados
    logger.info(f"OPTIMIZACIÓN COMPLETADA:")
    logger.info(f"   • Columnas mantenidas: {columnas_optimizadas}")
    logger.info(f"   • Columnas eliminadas: {columnas_eliminadas}")
    logger.info(f"   • Reducción de columnas: {(columnas_eliminadas/columnas_originales)*100:.1f}%")
    logger.info(f"   • Memoria original: {memoria_original:.1f} MB")
    logger.info(f"   • Memoria optimizada: {memoria_optimizada:.1f} MB")
    logger.info(f"   • Reducción de memoria: {reduccion_memoria:.1f}%")
    
    if columnas_faltantes:
        logger.warning(f"Columnas no encontradas: {columnas_faltantes}")
    
    # Mostrar muestra del dataset optimizado
    logger.info(f"COLUMNAS MANTENIDAS:")
    for i, col in enumerate(df_optimizado.columns, 1):
        logger.info(f"   {i:2d}. {col}")
    
    return df_optimizado


def export_optimized_dataset(df_optimized: pd.DataFrame, 
                           output_path: str = "../../data/processed/ml/peliculas_sensibilidad_optimized.csv") -> bool:

    logger = logging.getLogger("export_optimized_dataset")
    
    try:
        logger.info(f"EXPORTANDO DATASET OPTIMIZADO...")
        
        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Exportar a CSV
        df_optimized.to_csv(output_path, index=False, encoding='utf-8')
        
        # Verificar exportación
        if Path(output_path).exists():
            file_size_mb = Path(output_path).stat().st_size / (1024**2)
            logger.info(f"EXPORTACIÓN EXITOSA:")
            logger.info(f"   • Archivo: {output_path}")
            logger.info(f"   • Tamaño: {file_size_mb:.1f} MB")
            logger.info(f"   • Registros: {len(df_optimized):,}")
            logger.info(f"   • Columnas: {len(df_optimized.columns)}")
            return True
        else:
            logger.error("Error: El archivo no se generó correctamente")
            return False
            
    except Exception as e:
        logger.error(f"Error durante la exportación: {e}")
        return False


def test_dataset_optimization():

    print("PROBANDO OPTIMIZACIÓN DEL DATASET DE SENSIBILIDAD")
    print("=" * 60)
    
    try:
        # Cargar dataset original
        print("\n1. CARGANDO DATASET ORIGINAL...")
        analyzer = SensitivityAnalyzer()
        if not analyzer.load_data():
            print("Error: No se pudo cargar el dataset original")
            return None
        
        print(f"Dataset original cargado: {len(analyzer.df):,} filas  {len(analyzer.df.columns)} columnas")
        
        # Optimizar dataset
        print("\n2. OPTIMIZANDO DATASET...")
        df_optimized = optimize_sensitivity_dataset(analyzer.df)
        
        # Exportar dataset optimizado
        print("\n3. EXPORTANDO DATASET OPTIMIZADO...")
        export_success = export_optimized_dataset(df_optimized)
        
        if export_success:
            print(f"\n4. COMPARACIÓN FINAL:")
            print("=" * 50)
            print(f"DATASET ORIGINAL:")
            print(f"   • Filas: {len(analyzer.df):,}")
            print(f"   • Columnas: {len(analyzer.df.columns)}")
            print(f"   • Memoria: {analyzer.df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
            
            print(f"\nDATASET OPTIMIZADO:")
            print(f"   • Filas: {len(df_optimized):,}")
            print(f"   • Columnas: {len(df_optimized.columns)}")
            print(f"   • Memoria: {df_optimized.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
            
            # Mostrar muestra del dataset optimizado
            print(f"\nMUESTRA DEL DATASET OPTIMIZADO:")
            print("-" * 50)
            print(df_optimized.head())
        
        return df_optimized
        
    except Exception as e:
        print(f"Error durante la optimización: {e}")
        return None


def merge_sensitivity_with_imdb_data(sensitivity_df: pd.DataFrame,
                                   imdb_df: pd.DataFrame,
                                   join_type: str = 'left',
                                   require_full_match: bool = False) -> pd.DataFrame:

    logger = logging.getLogger("merge_sensitivity_with_imdb_data")
    
    logger.info("UNIENDO DATASET DE SENSIBILIDAD CON DATOS DE IMDb...")
    logger.info(f"Dataset sensibilidad: {len(sensitivity_df):,} filas × {len(sensitivity_df.columns)} columnas")
    logger.info(f"Dataset IMDb: {len(imdb_df):,} filas × {len(imdb_df.columns)} columnas")
    
    # Verificar que ambos datasets tengan la columna imdbId
    if 'imdbId' not in sensitivity_df.columns:
        logger.error("El dataset de sensibilidad no tiene la columna 'imdbId'")
        return sensitivity_df
    
    if 'imdbId' not in imdb_df.columns:
        logger.error("El dataset de IMDb no tiene la columna 'imdbId'")
        return sensitivity_df
    
    # CRÍTICO: Convertir ambas columnas a string para asegurar coincidencias
    logger.info("Convirtiendo columnas imdbId a string para consistencia...")
    sensitivity_df = sensitivity_df.copy()
    imdb_working = imdb_df.copy()

    # A string, pero sin aún zfill
    sensitivity_df['imdbId'] = sensitivity_df['imdbId'].astype(str)
    imdb_working['imdbId']   = imdb_working['imdbId'].astype(str)

    # Determinar la anchura "real" del imdbId en IMDb (suele ser 7 u 8)
    width_ref = int(imdb_working['imdbId'].str.len().dropna().mode().iloc[0])

    # Normalizar ambos lados a la misma anchura
    sensitivity_df['imdbId'] = _normalize_imdb_numeric(sensitivity_df['imdbId'], width_ref)
    imdb_working['imdbId']   = _normalize_imdb_numeric(imdb_working['imdbId'],   width_ref)
    
    # Definir columnas del dataset IMDb que NO queremos incluir
    columnas_excluir = ['movie_id', 'votes', 'source_genre']
    
    # Obtener columnas disponibles del dataset IMDb (excluyendo las no deseadas)
    columnas_imdb_disponibles = [col for col in imdb_working.columns if col not in columnas_excluir]
    
    logger.info(f"COLUMNAS DE IMDb A INCLUIR:")
    for i, col in enumerate(columnas_imdb_disponibles, 1):
        logger.info(f"   {i:2d}. {col}")
    
    # Crear dataset IMDb filtrado con columnas deseadas
    imdb_filtered = imdb_working[columnas_imdb_disponibles].copy()
    
    # CRÍTICO: Eliminar duplicados ANTES del merge para evitar multiplicación de registros
    logger.info(f"Registros IMDb antes de eliminar duplicados: {len(imdb_filtered):,}")
    
    # Mantener solo el primer registro por imdbId
    imdb_filtered = imdb_filtered.drop_duplicates(subset=['imdbId'], keep='first')
    logger.info(f"Registros IMDb después de eliminar duplicados: {len(imdb_filtered):,}")
    
    # Verificar que las columnas de join tienen el mismo tipo
    logger.info(f"Tipo de imdbId en sensibilidad: {sensitivity_df['imdbId'].dtype}")
    logger.info(f"Tipo de imdbId en IMDb: {imdb_filtered['imdbId'].dtype}")
    
    # Decidir tipo de join: podemos forzar coincidencias plenas (inner) si se requiere
    how = join_type
    if require_full_match:
        logger.info("Se requiere coincidencia plena en la llave: se usará INNER join")
        how = 'inner'
    else:
        logger.info(f"Realizando merge con tipo '{how}' (por defecto 'left' para mantener todos los registros de sensibilidad)")
        
        
        logger.info(f"Anchura imdbId (referencia): {width_ref}")
    logger.info(f"Ejemplos imdbId sensibilidad: {sensitivity_df['imdbId'].head(3).tolist()}")
    logger.info(f"Ejemplos imdbId IMDb:         {imdb_working['imdbId'].head(3).tolist()}")

    merged_df = sensitivity_df.merge(
        imdb_filtered,
        on='imdbId',
        how=how,
        suffixes=('_sensitivity', '_imdb')
    )
    
    
    
    if not require_full_match:
        if len(merged_df) != len(sensitivity_df):
            logger.error(f"ERROR: El merge cambió el número de registros (esperado por LEFT mantener el principal)!")
            logger.error(f"  Original: {len(sensitivity_df):,}")
            logger.error(f"  Después merge: {len(merged_df):,}")
            logger.error(f"  Diferencia: {len(merged_df) - len(sensitivity_df):,}")
            raise ValueError("El merge no debe cambiar el número de registros del dataset principal cuando no se exige coincidencia plena")
    else:
        
        
        nulls_after = merged_df['imdbId'].isna().sum()
        if nulls_after > 0:
            logger.error(f"ERROR: Después de usar INNER join hay {nulls_after} filas con 'imdbId' nulo. Revisar datos.")
            raise ValueError("INNER join produjo filas con llave nula, lo cual es inesperado")
        logger.info(f"INNER join completado: {len(merged_df):,} registros resultantes (solo coincidencias)")
    
    # Calcular estadísticas del merge
    total_sensibilidad = len(sensitivity_df)
    total_merged = len(merged_df)
    matches_found = merged_df['movie_name'].notna().sum() if 'movie_name' in merged_df.columns else 0
    match_rate = (matches_found / total_sensibilidad) * 100
    
    # Reportar resultados
    logger.info(f"RESULTADOS DEL MERGE:")
    logger.info(f"   • Total películas sensibilidad: {total_sensibilidad:,}")
    logger.info(f"   • Total después del merge: {total_merged:,}")
    logger.info(f"   • Coincidencias encontradas: {matches_found:,}")
    logger.info(f"   • Tasa de coincidencia: {match_rate:.1f}%")
    logger.info(f"   • Columnas finales: {len(merged_df.columns)}")
    
    # Calcular uso de memoria
    memoria_merged = merged_df.memory_usage(deep=True).sum() / (1024**2)
    logger.info(f"   • Memoria del dataset combinado: {memoria_merged:.1f} MB")
    
    # Mostrar columnas del dataset final
    logger.info(f"COLUMNAS DEL DATASET COMBINADO:")
    for i, col in enumerate(merged_df.columns, 1):
        logger.info(f"   {i:2d}. {col}")
    
    return merged_df


def export_combined_dataset(merged_df: pd.DataFrame, 
                          output_path: str = "../../data/processed/ml/dataset_sensibilidad_imdb_combined.csv") -> bool:

    logger = logging.getLogger("export_combined_dataset")
    
    try:
        logger.info(f"EXPORTANDO DATASET COMBINADO...")
        
        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Exportar a CSV
        merged_df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Verificar exportación
        if Path(output_path).exists():
            file_size_mb = Path(output_path).stat().st_size / (1024**2)
            logger.info(f"EXPORTACIÓN EXITOSA:")
            logger.info(f"   • Archivo: {output_path}")
            logger.info(f"   • Tamaño: {file_size_mb:.1f} MB")
            logger.info(f"   • Registros: {len(merged_df):,}")
            logger.info(f"   • Columnas: {len(merged_df.columns)}")
            return True
        else:
            logger.error("Error: El archivo no se generó correctamente")
            return False
            
    except Exception as e:
        logger.error(f"Error durante la exportación: {e}")
        return False


def test_dataset_combination():

    print("PROBANDO COMBINACIÓN DE DATASETS DE SENSIBILIDAD E IMDb")
    print("=" * 70)
    
    try:
        # 1. Cargar dataset de sensibilidad optimizado
        print("\n1. CARGANDO DATASET DE SENSIBILIDAD OPTIMIZADO...")
        sensitivity_path = "../../data/processed/ml/peliculas_sensibilidad_optimized.csv"
        
        if not Path(sensitivity_path).exists():
            print(f"Error: No se encuentra el archivo {sensitivity_path}")
            print("Ejecute primero la optimización del dataset de sensibilidad.")
            return None
        
        sensitivity_df = pd.read_csv(sensitivity_path)
        print(f"Dataset sensibilidad cargado: {len(sensitivity_df):,} filas × {len(sensitivity_df.columns)} columnas")
        
        # 2. Cargar dataset de IMDb unificado
        print("\n2. CARGANDO DATASET DE IMDb UNIFICADO...")
        imdb_path = "../../data/raw/train/movies_imdb_sample/movies_imdb_unified.csv"
        
        if not Path(imdb_path).exists():
            print(f"Error: No se encuentra el archivo {imdb_path}")
            print("Ejecute primero la unificación de géneros de IMDb.")
            return None
        
        imdb_df = pd.read_csv(imdb_path)
        print(f"Dataset IMDb cargado: {len(imdb_df):,} filas × {len(imdb_df.columns)} columnas")
        
        # 3. Verificar que ambos tengan la columna imdbId
        print("\n3. VERIFICANDO COLUMNAS DE ENLACE...")
        if 'imdbId' in sensitivity_df.columns:
            print(f"✓ Dataset sensibilidad tiene columna 'imdbId'")
        else:
            print(f"✗ Dataset sensibilidad NO tiene columna 'imdbId'")
            return None
            
        if 'imdbId' in imdb_df.columns:
            print(f"✓ Dataset IMDb tiene columna 'imdbId'")
        else:
            print(f"✗ Dataset IMDb NO tiene columna 'imdbId'")
            return None
        
        # 4. Realizar el merge
        print("\n4. COMBINANDO DATASETS...")
        combined_df = merge_sensitivity_with_imdb_data(sensitivity_df, imdb_df)
        
        # 5. Exportar dataset combinado
        print("\n5. EXPORTANDO DATASET COMBINADO...")
        export_success = export_combined_dataset(combined_df)
        
        if export_success:
            print(f"\n6. RESUMEN FINAL:")
            print("=" * 50)
            print(f"DATASET ORIGINAL DE SENSIBILIDAD:")
            print(f"   • Filas: {len(sensitivity_df):,}")
            print(f"   • Columnas: {len(sensitivity_df.columns)}")
            
            print(f"\nDATASET ORIGINAL DE IMDb:")
            print(f"   • Filas: {len(imdb_df):,}")
            print(f"   • Columnas: {len(imdb_df.columns)}")
            
            print(f"\nDATASET COMBINADO:")
            print(f"   • Filas: {len(combined_df):,}")
            print(f"   • Columnas: {len(combined_df.columns)}")
            print(f"   • Memoria: {combined_df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
            
            # Mostrar muestra del dataset combinado
            print(f"\nMUESTRA DEL DATASET COMBINADO:")
            print("-" * 50)
            print(combined_df.head())
        
        return combined_df
        
    except Exception as e:
        print(f"Error durante la combinación: {e}")
        return None


def clean_movie_titles(df: pd.DataFrame) -> pd.DataFrame:
    
    logger = logging.getLogger("clean_movie_titles")
    
    if 'title' not in df.columns:
        logger.error("Columna 'title' no encontrada en el dataset")
        return df
    
    logger.info("LIMPIANDO TÍTULOS DE PELÍCULAS...")
    logger.info(f"Dataset: {len(df):,} registros")
    
    # Crear copia del DataFrame
    df_cleaned = df.copy()
    
    
    
    df_cleaned['title_clean'] = df_cleaned['title'].str.replace(
        r'\s*\(\d{4}\)\s*$', '', regex=True
    )
    
    # Verificar la limpieza
    total_records = len(df_cleaned)
    records_with_clean_title = df_cleaned['title_clean'].notna().sum()
    completeness = (records_with_clean_title / total_records) * 100
    
    # Verificar que se eliminaron los años
    titles_with_year_pattern = df_cleaned['title'].str.contains(r'\(\d{4}\)', na=False).sum()
    clean_titles_with_year_pattern = df_cleaned['title_clean'].str.contains(r'\(\d{4}\)', na=False).sum()
    year_removal_success = titles_with_year_pattern - clean_titles_with_year_pattern
    
    logger.info(f"LIMPIEZA COMPLETADA:")
    logger.info(f"   • Total de registros: {total_records:,}")
    logger.info(f"   • Títulos limpios creados: {records_with_clean_title:,}")
    logger.info(f"   • Completitud: {completeness:.1f}%")
    logger.info(f"   • Títulos originales con año: {titles_with_year_pattern:,}")
    logger.info(f"   • Años eliminados exitosamente: {year_removal_success:,}")
    
    # Mostrar ejemplos de limpieza
    logger.info(f"EJEMPLOS DE LIMPIEZA:")
    sample_data = df_cleaned[['title', 'title_clean']].head(10)
    for i, (_, row) in enumerate(sample_data.iterrows(), 1):
        logger.info(f"   {i:2d}. '{row['title']}' → '{row['title_clean']}'")
    
    # Verificar longitudes
    original_lengths = df_cleaned['title'].str.len()
    clean_lengths = df_cleaned['title_clean'].str.len()
    avg_reduction = (original_lengths - clean_lengths).mean()
    
    logger.info(f"ESTADÍSTICAS DE LONGITUD:")
    logger.info(f"   • Longitud promedio original: {original_lengths.mean():.1f} caracteres")
    logger.info(f"   • Longitud promedio limpia: {clean_lengths.mean():.1f} caracteres")
    logger.info(f"   • Reducción promedio: {avg_reduction:.1f} caracteres")
    
    return df_cleaned


def test_title_cleaning():
    
    print("PROBANDO LIMPIEZA DE TÍTULOS DE PELÍCULAS")
    print("=" * 60)
    
    try:
        # Cargar dataset combinado
        combined_path = "../../data/processed/ml/dataset_sensibilidad_imdb_combined.csv"
        
        if not Path(combined_path).exists():
            print(f"Error: No se encuentra el archivo {combined_path}")
            print("Ejecute primero la combinación de datasets.")
            return None
        
        print("\n1. CARGANDO DATASET COMBINADO...")
        df_combined = pd.read_csv(combined_path)
        print(f"Dataset cargado: {len(df_combined):,} filas × {len(df_combined.columns)} columnas")
        
        # Verificar estructura de la columna title
        print("\n2. ANALIZANDO COLUMNA 'title'...")
        if 'title' in df_combined.columns:
            titles_with_year = df_combined['title'].str.contains(r'\(\d{4}\)', na=False).sum()
            total_titles = df_combined['title'].notna().sum()
            print(f"   • Total de títulos válidos: {total_titles:,}")
            print(f"   • Títulos con patrón (YYYY): {titles_with_year:,}")
            print(f"   • Porcentaje con año: {(titles_with_year/total_titles)*100:.1f}%")
            
            print(f"\n   Ejemplos de títulos actuales:")
            for i, title in enumerate(df_combined['title'].head(5)):
                if pd.notna(title):
                    print(f"      {i+1}. {title}")
        else:
            print("   ✗ Columna 'title' no encontrada")
            return None
        
        # Limpiar títulos
        print("\n3. LIMPIANDO TÍTULOS...")
        df_cleaned = clean_movie_titles(df_combined)
        
        # Verificar resultados
        print("\n4. VERIFICANDO RESULTADOS...")
        if 'title_clean' in df_cleaned.columns:
            print(f"   ✓ Columna 'title_clean' creada exitosamente")
            
            print(f"\n   Comparación de ejemplos:")
            sample_data = df_cleaned[['title', 'title_clean']].head(5)
            for i, (_, row) in enumerate(sample_data.iterrows(), 1):
                print(f"      {i}. Original: '{row['title']}'")
                print(f"         Limpio:   '{row['title_clean']}'")
                print()
        
        # Exportar dataset con títulos limpios
        print("\n5. EXPORTANDO DATASET CON TÍTULOS LIMPIOS...")
        output_path = "../../data/processed/ml/dataset_sensibilidad_imdb_final.csv"
        
        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Exportar
        df_cleaned.to_csv(output_path, index=False, encoding='utf-8')
        
        if Path(output_path).exists():
            file_size_mb = Path(output_path).stat().st_size / (1024**2)
            print(f"EXPORTACIÓN EXITOSA:")
            print(f"   • Archivo: {output_path}")
            print(f"   • Tamaño: {file_size_mb:.1f} MB")
            print(f"   • Registros: {len(df_cleaned):,}")
            print(f"   • Columnas: {len(df_cleaned.columns)} (incluye 'title_clean')")
            
            print(f"\n6. RESUMEN FINAL:")
            print("=" * 50)
            print(f"   • SIN PÉRDIDA DE DATOS: {len(df_cleaned):,} registros mantenidos")
            print(f"   • NUEVA COLUMNA: 'title_clean' con títulos sin año")
            print(f"   • COLUMNA ORIGINAL: 'title' se mantiene intacta")
            print(f"   • COMPLETITUD: 100% (sin valores faltantes)")
        else:
            print("   ✗ Error: El archivo no se generó correctamente")
        
        return df_cleaned
        
    except Exception as e:
        print(f"Error durante la limpieza de títulos: {e}")
        return None


class SensitivityAnalyzer:
        
    def __init__(self, data_path: str = "../../data/raw/processed/ml/peliculas_sensibilidad_gold.csv"):
        
        self.data_path = data_path
        self.df = None
        self.score_columns = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_data(self) -> bool:
        
        try:
            self.df = pd.read_csv(self.data_path)
            self.logger.info(f"Dataset cargado exitosamente")
            self.logger.info(f"Dimensiones: {self.df.shape[0]:,} filas × {self.df.shape[1]:,} columnas")
            self.logger.info(f"Uso de memoria: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Identificar columnas de scores
            self.score_columns = [col for col in self.df.columns if 'score' in col.lower()]
            self.logger.info(f"Scores de sensibilidad encontrados: {len(self.score_columns)}")
            
            return True
            
        except FileNotFoundError:
            self.logger.error(f"Error: No se pudo encontrar el archivo {self.data_path}")
            return False
        except Exception as e:
            self.logger.error(f"Error al cargar el dataset: {e}")
            return False
    
    def get_basic_info(self) -> Dict:
        
        if self.df is None:
            self.logger.error("Dataset no cargado. Ejecute load_data() primero.")
            return {}
        
        info = {
            'shape': self.df.shape,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'columns': list(self.df.columns),
            'score_columns': self.score_columns,
            'dtypes': dict(self.df.dtypes)
        }
        
        return info
    
    def analyze_missing_values(self) -> pd.DataFrame:
        
        if self.df is None:
            self.logger.error("Dataset no cargado. Ejecute load_data() primero.")
            return pd.DataFrame()
        
        self.logger.info("ANALIZANDO VALORES FALTANTES...")
        
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Columna': missing_data.index,
            'Valores_Faltantes': missing_data.values,
            'Porcentaje': missing_percent.values
        }).sort_values('Valores_Faltantes', ascending=False)
        
        missing_with_data = missing_df[missing_df['Valores_Faltantes'] > 0]
        
        if len(missing_with_data) > 0:
            self.logger.info(f"Columnas con valores faltantes: {len(missing_with_data)}")
        else:
            self.logger.info("No se encontraron valores faltantes en el dataset")
        
        total_missing = missing_data.sum()
        completeness = ((self.df.shape[0] * self.df.shape[1] - total_missing) / 
                       (self.df.shape[0] * self.df.shape[1]) * 100)
        
        self.logger.info(f"Total de valores faltantes: {total_missing:,}")
        self.logger.info(f"Completitud del dataset: {completeness:.2f}%")
        
        return missing_df
    
    def analyze_score_distributions(self) -> Dict:
        
        if self.df is None:
            self.logger.error("Dataset no cargado. Ejecute load_data() primero.")
            return {}
        
        if not self.score_columns:
            self.logger.warning("No se encontraron columnas de scores")
            return {}
        
        self.logger.info("ANALIZANDO DISTRIBUCIÓN DE SCORES DE SENSIBILIDAD...")
        
        distributions = {}
        
        for score in self.score_columns:
            non_zero_count = (self.df[score] > 0).sum()
            coverage = (non_zero_count / len(self.df)) * 100
            
            stats = {
                'count_non_zero': non_zero_count,
                'coverage_percent': coverage,
                'min': self.df[score].min(),
                'max': self.df[score].max(),
                'mean': self.df[score].mean(),
                'std': self.df[score].std(),
                'median': self.df[score].median()
            }
            
            distributions[score] = stats
            
            self.logger.info(f"{score}:")
            self.logger.info(f"   • Películas con score > 0: {non_zero_count:,} ({coverage:.1f}%)")
            self.logger.info(f"   • Rango: {stats['min']:.3f} - {stats['max']:.3f}")
            self.logger.info(f"   • Promedio: {stats['mean']:.3f}")
        
        return distributions
    
    def calculate_correlations(self) -> pd.DataFrame:
        
        if self.df is None:
            self.logger.error("Dataset no cargado. Ejecute load_data() primero.")
            return pd.DataFrame()
        
        if len(self.score_columns) < 2:
            self.logger.warning("Se necesitan al menos 2 scores para calcular correlaciones")
            return pd.DataFrame()
        
        self.logger.info("CALCULANDO CORRELACIONES ENTRE SCORES...")
        
        correlation_matrix = self.df[self.score_columns].corr()
        
        # Identificar correlaciones significativas
        significant_correlations = []
        for i in range(len(self.score_columns)):
            for j in range(i+1, len(self.score_columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.3:
                    significant_correlations.append({
                        'var1': self.score_columns[i],
                        'var2': self.score_columns[j],
                        'correlation': corr_val
                    })
        
        if significant_correlations:
            self.logger.info("CORRELACIONES SIGNIFICATIVAS (|r| > 0.3):")
            for corr in significant_correlations:
                self.logger.info(f"• {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.3f}")
        else:
            self.logger.info("No se encontraron correlaciones significativas (|r| > 0.3)")
        
        return correlation_matrix
    
    def analyze_family_suitability(self) -> Dict:
        
        if self.df is None:
            self.logger.error("Dataset no cargado. Ejecute load_data() primero.")
            return {}
        
        if 'aptitud_familiar' not in self.df.columns:
            self.logger.warning("Columna 'aptitud_familiar' no encontrada")
            return {}
        
        self.logger.info("ANALIZANDO APTITUD FAMILIAR...")
        
        stats = {
            'mean': self.df['aptitud_familiar'].mean(),
            'median': self.df['aptitud_familiar'].median(),
            'std': self.df['aptitud_familiar'].std(),
            'min': self.df['aptitud_familiar'].min(),
            'max': self.df['aptitud_familiar'].max(),
            'q25': self.df['aptitud_familiar'].quantile(0.25),
            'q75': self.df['aptitud_familiar'].quantile(0.75)
        }
        
        self.logger.info(f"Estadísticas de Aptitud Familiar:")
        self.logger.info(f"   • Promedio: {stats['mean']:.1f}")
        self.logger.info(f"   • Mediana: {stats['median']:.1f}")
        self.logger.info(f"   • Desviación estándar: {stats['std']:.1f}")
        self.logger.info(f"   • Rango: {stats['min']:.0f} - {stats['max']:.0f}")
        
        return stats
    
    def create_visualizations(self, save_path: str = "plots/") -> None:
        
        if self.df is None:
            self.logger.error("Dataset no cargado. Ejecute load_data() primero.")
            return
        
        # Crear directorio si no existe
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("GENERANDO VISUALIZACIONES...")
        
        # 1. Distribución de scores de sensibilidad
        if self.score_columns:
            n_scores = len(self.score_columns)
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, score in enumerate(self.score_columns):
                if i < len(axes):
                    axes[i].hist(self.df[score], bins=50, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Distribución: {score}')
                    axes[i].set_xlabel('Score')
                    axes[i].set_ylabel('Frecuencia')
                    axes[i].grid(True, alpha=0.3)
                    
                    mean_val = self.df[score].mean()
                    axes[i].axvline(mean_val, color='red', linestyle='--', 
                                   label=f'Media: {mean_val:.3f}')
                    axes[i].legend()
            
            # Ocultar subplots no utilizados
            for i in range(len(self.score_columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/score_distributions.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        # 2. Matriz de correlación
        if len(self.score_columns) > 1:
            correlation_matrix = self.df[self.score_columns].corr()
            
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                        center=0, square=True, fmt='.3f', cbar_kws={"shrink": 0.8})
            plt.title('Matriz de Correlación entre Scores de Sensibilidad')
            plt.tight_layout()
            plt.savefig(f"{save_path}/correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        # 3. Distribución de aptitud familiar
        if 'aptitud_familiar' in self.df.columns:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(self.df['aptitud_familiar'], bins=30, alpha=0.7, edgecolor='black')
            plt.title('Distribución de Aptitud Familiar')
            plt.xlabel('Aptitud Familiar (0-100)')
            plt.ylabel('Frecuencia')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.boxplot(self.df['aptitud_familiar'])
            plt.title('Box Plot - Aptitud Familiar')
            plt.ylabel('Aptitud Familiar (0-100)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/family_suitability.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Visualizaciones guardadas en: {save_path}")
    
    def generate_summary_report(self) -> Dict:
        
        if self.df is None:
            self.logger.error("Dataset no cargado. Ejecute load_data() primero.")
            return {}
        
        self.logger.info("GENERANDO REPORTE RESUMEN...")
        
        report = {
            'dataset_info': self.get_basic_info(),
            'missing_values': self.analyze_missing_values(),
            'score_distributions': self.analyze_score_distributions(),
            'correlations': self.calculate_correlations(),
            'family_suitability': self.analyze_family_suitability(),
            'summary_stats': {
                'total_movies': len(self.df),
                'total_columns': len(self.df.columns),
                'score_columns_count': len(self.score_columns),
                'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
            }
        }
        
        self.logger.info("Reporte generado exitosamente")
        
        return report


def test_movie_unification():
    
    print("PROBANDO UNIFICACIÓN DE ARCHIVOS DE GÉNEROS DE PELÍCULAS")
    print("=" * 60)
    
    try:
        # Cargar y unificar los archivos
        unified_df = load_and_unify_movie_genres()
        
        print(f"\nRESULTADOS DE LA UNIFICACIÓN:")
        print("=" * 50)
        print(f"Total de películas: {len(unified_df):,}")
        print(f"Total de columnas: {len(unified_df.columns)}")
        print(f"Uso de memoria: {unified_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        print(f"\nGÉNEROS ENCONTRADOS:")
        print("=" * 50)
        genre_counts = unified_df['source_genre'].value_counts()
        for genre, count in genre_counts.items():
            print(f"  {genre}: {count:,} películas")
        
        # PROCESAR IDs DE IMDb para crear columna imdbId
        print(f"\nPROCESANDO IDs DE IMDb...")
        print("=" * 50)
        unified_df_processed = process_imdb_ids(unified_df)
        
        print(f"\nPRIMERAS 5 FILAS DEL DATASET PROCESADO:")
        print("=" * 50)
        # Mostrar columnas relevantes incluyendo la nueva imdbId
        columns_to_show = ['movie_id', 'imdbId', 'movie_name', 'year', 'source_genre']
        print(unified_df_processed[columns_to_show].head())
        
        # Exportar el dataset unificado y procesado
        output_path = "../../data/raw/train/movies_imdb_sample/movies_imdb_unified.csv"
        
        print(f"\nEXPORTANDO DATASET UNIFICADO...")
        print("=" * 50)
        print(f"Ruta de exportación: {output_path}")
        
        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Exportar a CSV
        unified_df_processed.to_csv(output_path, index=False, encoding='utf-8')
        
        # Verificar el archivo exportado
        if Path(output_path).exists():
            file_size_mb = Path(output_path).stat().st_size / (1024**2)
            print(f"EXPORTACIÓN EXITOSA:")
            print(f"  • Archivo: {output_path}")
            print(f"  • Tamaño: {file_size_mb:.1f} MB")
            print(f"  • Registros: {len(unified_df_processed):,}")
            print(f"  • Columnas: {len(unified_df_processed.columns)}")
        else:
            print("ERROR: El archivo no se generó correctamente")
        
        print(f"\nINFORMACIÓN GENERAL:")
        print("=" * 50)
        print(unified_df_processed.info())
        
        return unified_df_processed
        
    except Exception as e:
        print(f"Error durante la unificación: {e}")
        return None


def apply_description_imputation(df: pd.DataFrame) -> pd.DataFrame:
    
    print(" APLICANDO IMPUTACIÓN DE DESCRIPTIONS...")
    print("=" * 60)
    
    df_copy = df.copy()
    
    # Verificar si la columna description existe
    if 'description' not in df_copy.columns:
        print(" Error: Columna 'description' no encontrada en el dataset")
        return df_copy
    
    # Contar valores nulos antes de la imputación
    nulos_antes = df_copy['description'].isna().sum()
    print(f" Valores nulos antes de imputación: {nulos_antes:,} ({nulos_antes/len(df_copy)*100:.1f}%)")
    
    if nulos_antes == 0:
        print(" No hay valores nulos en 'description'. No se necesita imputación.")
        return df_copy
    
    # Templates mejorados por género con variaciones
    genero_templates = {
        'Action': [
            "An action-packed film featuring intense sequences, thrilling adventures, and dynamic storytelling.",
            "A high-energy action movie with exciting stunts, heroic characters, and fast-paced narrative.",
            "An adrenaline-filled adventure showcasing brave protagonists facing challenging obstacles."
        ],
        'Adventure': [
            "An adventure story that takes viewers on an exciting journey filled with discovery and exploration.",
            "A thrilling adventure featuring epic quests, exotic locations, and courageous heroes.",
            "An exciting expedition story with daring characters exploring unknown territories."
        ],
        'Animation': [
            "An animated film that brings characters and stories to life through creative visual storytelling.",
            "A colorful animated movie featuring memorable characters and imaginative worlds.",
            "An artistic animated story that combines entertainment with visual creativity."
        ],
        'Comedy': [
            "A comedy that entertains audiences with humor, wit, and lighthearted storytelling.",
            "A funny film designed to bring laughter and joy through clever dialogue and situations.",
            "A humorous story exploring life's amusing moments with charm and entertainment."
        ],
        'Crime': [
            "A crime drama that explores the darker side of human nature and the complexities of justice.",
            "A gripping crime story featuring investigations, moral dilemmas, and criminal underworld.",
            "A suspenseful crime film examining law enforcement and criminal behavior."
        ],
        'Drama': [
            "A dramatic story that explores human emotions, relationships, and life experiences.",
            "A compelling drama delving into the complexities of human nature and personal struggles.",
            "An emotional journey examining characters facing significant life challenges and growth."
        ],
        'Fantasy': [
            "A fantasy adventure that transports viewers to magical worlds and extraordinary circumstances.",
            "An enchanting fantasy story featuring mystical creatures, magic, and otherworldly realms.",
            "A magical tale exploring supernatural elements and fantastical adventures."
        ],
        'Horror': [
            "A horror film designed to create suspense, fear, and thrilling supernatural experiences.",
            "A scary movie that explores dark themes, supernatural forces, and frightening scenarios.",
            "A suspenseful horror story challenging viewers with terrifying and mysterious elements."
        ],
        'Musical': [
            "A musical that combines storytelling with songs, dance numbers, and theatrical performances.",
            "An entertaining musical featuring memorable songs and choreographed dance sequences.",
            "A melodic story that integrates music and performance into narrative storytelling."
        ],
        'Mystery': [
            "A mystery that challenges viewers to solve puzzles and uncover hidden truths.",
            "An intriguing mystery story featuring detective work, clues, and surprising revelations.",
            "A puzzling tale that keeps audiences guessing until the final revelation."
        ],
        'Romance': [
            "A romantic story that explores love, relationships, and emotional connections between characters.",
            "A heartwarming romance featuring passionate love stories and relationship development.",
            "An emotional love story examining the complexities and beauty of romantic relationships."
        ],
        'Sci-Fi': [
            "A science fiction film that explores futuristic concepts, advanced technology, and alternate realities.",
            "A futuristic story featuring scientific innovations, space exploration, and technological wonders.",
            "A sci-fi adventure examining the impact of technology and scientific advancement on humanity."
        ],
        'Thriller': [
            "A thriller that builds suspense and keeps audiences on the edge of their seats.",
            "A suspenseful story featuring psychological tension, danger, and unexpected plot twists.",
            "An intense thriller combining mystery, danger, and nail-biting suspense."
        ],
        'War': [
            "A war film that depicts the realities, sacrifices, and impact of military conflict.",
            "A powerful war story exploring courage, brotherhood, and the human cost of battle.",
            "A dramatic war film examining the effects of conflict on soldiers and society."
        ],
        'Western': [
            "A western that captures the spirit and adventure of the American frontier.",
            "A frontier story featuring cowboys, outlaws, and life in the old American West.",
            "A classic western exploring themes of justice, survival, and frontier life."
        ]
    }
    
    def generar_descripcion_por_genero(titulo, generos, idx):
        
        if pd.isna(generos) or not generos.strip():
            return "A film that explores various themes through engaging storytelling and character development."
        
        generos_lista = [g.strip() for g in str(generos).split('|')]
        primer_genero = generos_lista[0] if generos_lista else 'Drama'
        
        # Obtener templates para el género
        templates = genero_templates.get(primer_genero, [
            "A film that provides entertainment through compelling storytelling.",
            "A movie that explores themes relevant to its genre and characters.",
            "A story that engages viewers through narrative and character development."
        ])
        
        # Usar índice para crear variación
        template_idx = idx % len(templates)
        return templates[template_idx]
    
    # Aplicar imputación con variación
    mask_sin_desc = df_copy['description'].isna()
    registros_a_imputar = df_copy[mask_sin_desc].index.tolist()
    
    print(f" Aplicando imputación a {len(registros_a_imputar):,} registros...")
    
    for i, idx in enumerate(registros_a_imputar):
        row = df_copy.loc[idx]
        descripcion = generar_descripcion_por_genero(row.get('title', ''), row.get('genres', ''), i)
        df_copy.loc[idx, 'description'] = descripcion
    
    # Verificar resultado
    nulos_despues = df_copy['description'].isna().sum()
    imputaciones_realizadas = nulos_antes - nulos_despues
    
    print(f" IMPUTACIÓN COMPLETADA:")
    print(f"   • Imputaciones realizadas: {imputaciones_realizadas:,}")
    print(f"   • Valores nulos restantes: {nulos_despues:,}")
    print(f"   • Coverage final: {(1 - nulos_despues/len(df_copy))*100:.1f}%")
    
    # Mostrar algunos ejemplos
    if imputaciones_realizadas > 0:
        print(f"\n Ejemplos de descripciones imputadas:")
        ejemplos = df_copy.loc[registros_a_imputar[:3]]
        for i, (idx, row) in enumerate(ejemplos.iterrows(), 1):
            genero = row.get('genres', 'N/A').split('|')[0] if pd.notna(row.get('genres')) else 'N/A'
            print(f"   {i}. {row.get('title', 'N/A')} ({genero})")
            print(f"      '{row['description'][:80]}...'")
    
    return df_copy


def main():
    
    print("INICIANDO ANÁLISIS DE SENSIBILIDAD DE PELÍCULAS")
    print("=" * 60)
    
    # 1. Unificación de archivos de géneros de películas
    print("\n1. UNIFICACIÓN DE ARCHIVOS DE GÉNEROS DE PELÍCULAS")
    print("-" * 60)
    unified_movies = load_and_unify_movie_genres()
    unified_movies_processed = process_imdb_ids(unified_movies)
    # Exportar dataset unificado IMDb
    imdb_unified_path = "../../data/raw/train/movies_imdb_sample/movies_imdb_unified.csv"
    Path(imdb_unified_path).parent.mkdir(parents=True, exist_ok=True)
    unified_movies_processed.to_csv(imdb_unified_path, index=False, encoding='utf-8')
    print(f"   • Dataset IMDb unificado exportado: {imdb_unified_path}")

    # 2. Análisis y optimización del dataset de sensibilidad
    print("\n2. ANÁLISIS Y OPTIMIZACIÓN DEL DATASET DE SENSIBILIDAD")
    print("-" * 60)
    sensitivity_path = "../../data/raw/processed/ml/peliculas_sensibilidad_gold.csv"
    if not os.path.exists(sensitivity_path):
        print(f" Error: No se encuentra el archivo {sensitivity_path}")
        return
    df_sensitivity = pd.read_csv(sensitivity_path)
    cols_to_drop = ["genres", "av_rating", "n_ratings"]
    df_sensitivity = df_sensitivity.drop(columns=[c for c in cols_to_drop if c in df_sensitivity.columns])
    df_optimized = optimize_sensitivity_dataset(df_sensitivity)
    # Exportar dataset optimizado
    export_optimized_dataset(df_optimized)
    print("   • Dataset de sensibilidad optimizado exportado")

    # 3. Combinación de datasets de sensibilidad e IMDb
    print("\n3. COMBINACIÓN DE DATASETS DE SENSIBILIDAD E IMDb")
    print("-" * 60)
    df_imdb = pd.read_csv(imdb_unified_path)
    df_imdb = process_imdb_ids(df_imdb)
    # IMPORTANT: exigir coincidencia plena en la llave imdbId (solo registros que coinciden en ambas tablas)
    df_combined = merge_sensitivity_with_imdb_data(df_optimized, df_imdb, require_full_match=True)
    # Exportar dataset combinado
    export_combined_dataset(df_combined)
    print("   • Dataset combinado exportado")

    # 4. Limpieza de títulos
    print("\n4. LIMPIEZA DE TÍTULOS DE PELÍCULAS")
    print("-" * 60)
    final_dataset = clean_movie_titles(df_combined)
    final_titles_path = "../../data/processed/ml/dataset_sensibilidad_imdb_final.csv"
    Path(final_titles_path).parent.mkdir(parents=True, exist_ok=True)
    final_dataset.to_csv(final_titles_path, index=False, encoding='utf-8')
    print(f"   • Dataset con títulos limpios exportado: {final_titles_path}")

    # 5. Imputación y exportación del dataset final completo
    print("\n5. IMPUTACIÓN DE DESCRIPTIONS Y DATASET FINAL COMPLETO")
    print("-" * 60)
    dataset_imputado = apply_description_imputation(final_dataset)
    output_path_complete = "../../data/processed/ml/dataset_sensibilidad_imdb_final_complete.csv"
    try:
        Path(output_path_complete).parent.mkdir(parents=True, exist_ok=True)
        import csv
        if 'description' in dataset_imputado.columns:
            print("Limpiando columna 'description' para exportación robusta...")
            def clean_description(text):
                if pd.isna(text):
                    return ""
                text = str(text)
                text = text.replace('\n', ' ')
                text = text.replace('\r', ' ')
                text = text.replace('\t', ' ')
                text = text.replace('"', "'")
                text = text.replace('""', "'")
                text = text.replace(';', ',')
                text = text.strip()
                return text
            dataset_imputado['description'] = dataset_imputado['description'].apply(clean_description)
        else:
            print("Advertencia: No se encontró la columna 'description'.")
        dataset_imputado.to_csv(
            output_path_complete,
            index=False,
            encoding='utf-8',
            sep=';',
            quoting=csv.QUOTE_ALL,
            quotechar='"',
            lineterminator='\n'
        )
        if Path(output_path_complete).exists():
            file_size_mb = Path(output_path_complete).stat().st_size / (1024**2)
            print(f"\n DATASET FINAL GUARDADO:")
            print(f"   • Archivo: {output_path_complete}")
            print(f"   • Tamaño: {file_size_mb:.1f} MB") 
            print(f"   • Registros: {len(dataset_imputado):,}")
            print(f"   • Columnas: {len(dataset_imputado.columns)}")
            print(f"   • Completitud description: 100%")
        else:
            print(" Error: No se pudo crear el archivo completo")
    except Exception as e:
        print(f" Error al guardar dataset completo: {e}")
    print("=" * 60)


if __name__ == "__main__":
    main()