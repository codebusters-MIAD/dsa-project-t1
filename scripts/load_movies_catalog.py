#!/usr/bin/env python3
"""
Carga datos del CSV a movies_catalog
"""

import csv
import re
import sys
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values
from psycopg2 import sql


@dataclass
class LoadStats:
    """Stats de la carga"""
    total_read: int = 0
    inserted: int = 0
    updated: int = 0
    rejected: int = 0
    rejection_reasons: Dict[str, int] = None
    
    def __post_init__(self):
        if self.rejection_reasons is None:
            self.rejection_reasons = {}
    
    def add_rejection(self, reason: str):
        self.rejected += 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1


class MoviesCatalogLoader:
    """Loader para movies_catalog"""
    
    def __init__(self, db_url: str, batch_size: int = 100, dry_run: bool = False):
        self.db_url = db_url
        self.batch_size = batch_size
        self.dry_run = dry_run
        self.stats = LoadStats()
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Conectar a db"""
        try:
            self.conn = psycopg2.connect(self.db_url)
            self.cursor = self.conn.cursor()
            print(f"✓ Conectado a la base de datos")
        except Exception as e:
            print(f"✗ Error al conectar: {e}")
            raise
    
    def close(self):
        """Cerrar conexion"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
    
    def extract_runtime_minutes(self, runtime_str: str) -> Optional[int]:
        """Extraer minutos del runtime"""
        if not runtime_str or runtime_str.strip() == '':
            return None
        
        try:
            # patron horas y minutos
            hours_minutes = re.search(r'(\d+)h\s*(\d+)min', runtime_str)
            if hours_minutes:
                hours = int(hours_minutes.group(1))
                minutes = int(hours_minutes.group(2))
                total_minutes = hours * 60 + minutes
                return total_minutes if 0 < total_minutes < 500 else None
            
            # solo minutos
            minutes_only = re.search(r'(\d+)\s*min', runtime_str)
            if minutes_only:
                minutes = int(minutes_only.group(1))
                return minutes if 0 < minutes < 500 else None
            
            # solo numeros
            numbers = re.search(r'(\d+)', runtime_str)
            if numbers:
                minutes = int(numbers.group(1))
                return minutes if 0 < minutes < 500 else None
            
            return None
        except (ValueError, AttributeError):
            return None
    
    def parse_rating(self, rating_str: str) -> Optional[float]:
        """Parsear rating"""
        if not rating_str or rating_str.strip() == '':
            return None
        
        try:
            rating = float(rating_str)
            return rating if 0 <= rating <= 10 else None
        except (ValueError, TypeError):
            return None
    
    def parse_array(self, value: str) -> List[str]:
        """String separado por comas a array"""
        if not value or value.strip() == '':
            return []
        
        items = [item.strip() for item in value.split(',')]
        return [item for item in items if item]
    
    def format_imdb_id(self, imdb_id: str) -> Optional[str]:
        """Formatear imdb id con prefijo tt"""
        if not imdb_id or imdb_id.strip() == '':
            return None
        
        imdb_id = imdb_id.strip()
        if imdb_id.startswith('tt'):
            return imdb_id
        return f"tt{imdb_id}"
    
    def format_tmdb_id(self, tmdb_id: str) -> Optional[str]:
        """Formatear tmdb id sin decimales"""
        if not tmdb_id or tmdb_id.strip() == '':
            return None
        
        try:
            tmdb_num = float(tmdb_id)
            return str(int(tmdb_num))
        except (ValueError, TypeError):
            return None
    
    def validate_year(self, year: Optional[int]) -> bool:
        """Validar year"""
        if year is None:
            return True
        return 1800 <= year <= 2100
    
    def transform_row(self, row: Dict[str, str]) -> Optional[Dict]:
        """Transformar fila del CSV"""
        try:
            movie_name = row.get('movie_name', '').strip()
            if not movie_name:
                self.stats.add_rejection("movie_name vacio")
                return None
            
            # ids
            imdb_id = self.format_imdb_id(row.get('imdbId', ''))
            tmdb_id = self.format_tmdb_id(row.get('tmdbId', ''))
            
            if not imdb_id and not tmdb_id:
                self.stats.add_rejection("Sin ID valido")
                return None
            
            # year
            year_str = row.get('year', '').strip()
            year = int(year_str) if year_str and year_str.isdigit() else None
            if not self.validate_year(year):
                self.stats.add_rejection(f"Año invalido: {year}")
                return None
            
            runtime = self.extract_runtime_minutes(row.get('runtime', ''))
            rating = self.parse_rating(row.get('rating', ''))
            
            # arrays
            genre = self.parse_array(row.get('genre', ''))
            director = self.parse_array(row.get('director', ''))
            
            star_str = row.get('star', '').replace('\n', ' ').strip()
            star = self.parse_array(star_str)
            
            description = row.get('description', '').strip() or None
            
            transformed = {
                'imdb_id': imdb_id,
                'tmdb_id': tmdb_id,
                'movie_name': movie_name,
                'year': year,
                'runtime': runtime,
                'genre': genre if genre else None,
                'rating': rating,
                'description': description,
                'director': director if director else None,
                'star': star if star else None,
            }
            
            return transformed
            
        except Exception as e:
            self.stats.add_rejection(f"Error transformacion: {str(e)}")
            return None
    
    def upsert_batch(self, batch: List[Dict]) -> Tuple[int, int]:
        """Insertar o actualizar batch"""
        if not batch:
            return 0, 0
        
        values = []
        for record in batch:
            values.append((
                record['imdb_id'],
                record['tmdb_id'],
                record['movie_name'],
                record['year'],
                record['runtime'],
                record['genre'],
                record['rating'],
                record['description'],
                record['director'],
                record['star'],
            ))
        
        # upsert con tmdb_id
        upsert_query = """
            INSERT INTO movies_catalog (
                imdb_id, tmdb_id, movie_name, year, runtime,
                genre, rating, description, director, star
            ) VALUES %s
            ON CONFLICT (tmdb_id) 
            WHERE tmdb_id IS NOT NULL
            DO UPDATE SET
                imdb_id = EXCLUDED.imdb_id,
                movie_name = EXCLUDED.movie_name,
                year = EXCLUDED.year,
                runtime = EXCLUDED.runtime,
                genre = EXCLUDED.genre,
                rating = EXCLUDED.rating,
                description = EXCLUDED.description,
                director = EXCLUDED.director,
                star = EXCLUDED.star,
                updated_at = CURRENT_TIMESTAMP
        """
        
        # upsert con imdb_id
        upsert_query_imdb = """
            INSERT INTO movies_catalog (
                imdb_id, tmdb_id, movie_name, year, runtime,
                genre, rating, description, director, star
            ) VALUES %s
            ON CONFLICT (imdb_id) 
            WHERE imdb_id IS NOT NULL AND tmdb_id IS NULL
            DO UPDATE SET
                movie_name = EXCLUDED.movie_name,
                year = EXCLUDED.year,
                runtime = EXCLUDED.runtime,
                genre = EXCLUDED.genre,
                rating = EXCLUDED.rating,
                description = EXCLUDED.description,
                director = EXCLUDED.director,
                star = EXCLUDED.star,
                updated_at = CURRENT_TIMESTAMP
        """
        
        try:
            with_tmdb = [v for v in values if v[1] is not None]
            without_tmdb = [v for v in values if v[1] is None]
            
            inserted = 0
            updated = 0
            
            if with_tmdb:
                tmdb_ids = [v[1] for v in with_tmdb]
                self.cursor.execute(
                    "SELECT COUNT(*) FROM movies_catalog WHERE tmdb_id = ANY(%s)",
                    (tmdb_ids,)
                )
                existing_count = self.cursor.fetchone()[0]
                
                execute_values(self.cursor, upsert_query, with_tmdb)
                
                total_affected = len(with_tmdb)
                updated += existing_count
                inserted += (total_affected - existing_count)
            
            if without_tmdb:
                imdb_ids = [v[0] for v in without_tmdb]
                self.cursor.execute(
                    "SELECT COUNT(*) FROM movies_catalog WHERE imdb_id = ANY(%s) AND tmdb_id IS NULL",
                    (imdb_ids,)
                )
                existing_count = self.cursor.fetchone()[0]
                
                execute_values(self.cursor, upsert_query_imdb, without_tmdb)
                
                total_affected = len(without_tmdb)
                updated += existing_count
                inserted += (total_affected - existing_count)
            
            return inserted, updated
            
        except Exception as e:
            print(f"✗ Error en upsert: {e}")
            raise
    
    def load_csv(self, csv_path: str):
        """Cargar CSV"""
        print(f"\n{'='*60}")
        print(f"Iniciando carga de movies_catalog")
        print(f"{'='*60}")
        print(f"CSV: {csv_path}")
        print(f"DB: {self.db_url.split('@')[1] if '@' in self.db_url else 'local'}")
        print(f"Batch size: {self.batch_size}")
        print(f"Dry run: {self.dry_run}")
        print(f"{'='*60}\n")
        
        try:
            print("Leyendo CSV...")
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                
                batch = []
                
                for row in reader:
                    self.stats.total_read += 1
                    
                    transformed = self.transform_row(row)
                    if transformed is None:
                        continue
                    
                    batch.append(transformed)
                    
                    if len(batch) >= self.batch_size:
                        if not self.dry_run:
                            inserted, updated = self.upsert_batch(batch)
                            self.stats.inserted += inserted
                            self.stats.updated += updated
                            self.conn.commit()
                        else:
                            self.stats.inserted += len(batch)
                        
                        print(f"  Procesadas {self.stats.total_read} filas... "
                              f"(OK: {self.stats.inserted + self.stats.updated}, "
                              f"Rechazadas: {self.stats.rejected})")
                        batch = []
                
                # batch final
                if batch:
                    if not self.dry_run:
                        inserted, updated = self.upsert_batch(batch)
                        self.stats.inserted += inserted
                        self.stats.updated += updated
                        self.conn.commit()
                    else:
                        self.stats.inserted += len(batch)
                
                print(f"\n✓ CSV procesado\n")
                
        except FileNotFoundError:
            print(f"✗ Archivo no encontrado: {csv_path}")
            raise
        except Exception as e:
            print(f"✗ Error al procesar: {e}")
            if self.conn:
                self.conn.rollback()
            raise
    
    def print_summary(self):
        """Resumen de carga"""
        print(f"\n{'='*60}")
        print(f"RESUMEN DE CARGA")
        print(f"{'='*60}")
        print(f"Total filas leidas:     {self.stats.total_read:,}")
        print(f"Registros insertados:   {self.stats.inserted:,}")
        print(f"Registros actualizados: {self.stats.updated:,}")
        print(f"Registros rechazados:   {self.stats.rejected:,}")
        print(f"{'='*60}")
        
        if self.stats.rejection_reasons:
            print(f"\nMOTIVOS DE RECHAZO (Top 10):")
            print(f"{'-'*60}")
            sorted_reasons = sorted(
                self.stats.rejection_reasons.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            for reason, count in sorted_reasons:
                print(f"  {reason:50} {count:,}")
            print(f"{'-'*60}")
        
        if self.dry_run:
            print(f"\n⚠ DRY RUN: No se hicieron cambios en la db")
        
        print()


def main():
    parser = argparse.ArgumentParser(description='Cargar CSV a movies_catalog')
    
    parser.add_argument('--csv', required=True, help='Path al CSV')
    parser.add_argument('--db', required=True, help='DB url')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    parser.add_argument('--dry-run', action='store_true', help='Validar sin insertar')
    
    args = parser.parse_args()
    
    loader = MoviesCatalogLoader(
        db_url=args.db,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )
    
    try:
        loader.connect()
        loader.load_csv(args.csv)
        loader.print_summary()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n✗ Cancelado")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1
    finally:
        loader.close()


if __name__ == '__main__':
    sys.exit(main())
