"""Fixed article settings and raw input contract."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
_SHARED_ROOT=Path(__file__).resolve().parents[1]
if str(_SHARED_ROOT) not in sys.path:
    sys.path.insert(0,str(_SHARED_ROOT))
import runtime as rep
from shared_paths import workspace_root
import json
import numpy as np
HERE=Path(__file__).resolve().parent
DATA=(workspace_root()/'dataset/hnu_data/03_aal_roi_timeseries').resolve()
SCOPES=[('A_to_B','A','B'),('B_to_A','B','A'),('A_selected_all29','A',None),('B_selected_all29','B',None)]
KINDS=['local','global','pairwise','consensus','highgain']
MODES=['retest','temporal']
SEED=202610260
B=5000
def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))
def config():
    return read(HERE/'input_config.json')
def indices(rows,people,sessions):
    lookup={(r['subject'],r['order']):i for i,r in enumerate(rows)}
    return np.array([lookup[p,s] for p in people for s in sessions],dtype=np.int64)
def model(half,kind):
    key='strength' if kind in ['local','global'] else kind
    return read(HERE/'models.json')['models'][half][key]
