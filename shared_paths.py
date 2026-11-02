"""Common workspace paths; support the existing external dataset layout."""
from pathlib import Path
import os

MERGE_ROOT = Path(__file__).resolve().parent
def workspace_root():
    override = os.environ.get('FREEENERGY_WORKSPACE')
    return Path(override).expanduser().resolve() if override else MERGE_ROOT.parent.parent

def data_root():
    return workspace_root()/'dataset'
