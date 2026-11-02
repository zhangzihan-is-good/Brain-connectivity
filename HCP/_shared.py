"""Make the parent Merge_code modules importable for direct script entrypoints."""
from pathlib import Path
import sys

MERGE_ROOT = Path(__file__).resolve().parents[1]
if str(MERGE_ROOT) not in sys.path:
    sys.path.insert(0, str(MERGE_ROOT))
