#!/bin/bash
# wrapper para cargar movies_catalog

CSV_PATH="${1:-data/raw/train/consolidated/dataset_sensibilidad_imdb_final_complete.csv}"
DB_URL="${2:-postgresql://filmlens_user:filmlens_dev_2025@localhost:5435/filmlens}"
BATCH_SIZE="${3:-100}"
DRY_RUN="${4:-}"

echo "Carga de Movies Catalog - FilmLens"
echo ""

if [ ! -f "$CSV_PATH" ]; then
    echo "CSV no encontrado: $CSV_PATH"
    exit 1
fi

# check psycopg2
python3 -c "import psycopg2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "instalando psycopg2..."
    pip install psycopg2-binary
fi

CMD="python3 scripts/load_movies_catalog.py --csv \"$CSV_PATH\" --db \"$DB_URL\" --batch-size $BATCH_SIZE"

if [ "$DRY_RUN" == "--dry-run" ] || [ "$DRY_RUN" == "dry-run" ]; then
    CMD="$CMD --dry-run"
    echo "Modo DRY RUN - no se modificara la db"
    echo ""
fi

eval $CMD
