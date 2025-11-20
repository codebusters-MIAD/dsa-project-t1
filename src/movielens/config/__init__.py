"""
Configuracion del paquete movielens para clasificacion de sensibilidad.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

_CONFIG = None

def load_config() -> Dict[str, Any]:
    """Carga la configuracion desde config.yml"""
    global _CONFIG
    if _CONFIG is None:
        config_path = Path(__file__).parent / 'config.yml'
        with open(config_path, 'r', encoding='utf-8') as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG

config = load_config()
