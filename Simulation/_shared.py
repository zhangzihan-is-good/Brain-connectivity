"""Locate the common interfaces for direct script execution."""
from pathlib import Path
import sys
MERGE_ROOT=Path(__file__).resolve().parent.parent
if str(MERGE_ROOT) not in sys.path:sys.path.insert(0,str(MERGE_ROOT))
