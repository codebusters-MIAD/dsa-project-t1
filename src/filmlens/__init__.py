"""FilmLens - Multi-label trigger detection for movies."""

from pathlib import Path

# Read version
VERSION_FILE = Path(__file__).parent / "VERSION"
if VERSION_FILE.exists():
    with open(VERSION_FILE) as f:
        __version__ = f.read().strip()
else:
    __version__ = "0.1.0"
