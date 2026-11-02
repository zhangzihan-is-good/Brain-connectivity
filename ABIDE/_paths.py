"""Locations of the ABIDE implementation modules."""
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
CORE = BASE / 'core'
MODELS = BASE / 'models'
SUPPLEMENTS = BASE / 'supplements'


def activate():
    """Resolve the bundled modules independently of the working directory."""
    for directory in (BASE.parent, BASE, CORE, MODELS, SUPPLEMENTS):
        value = str(directory)
        if value not in sys.path:
            sys.path.insert(0, value)
