"""Native-estimator modules for all-participant robustness analyses."""
from __future__ import annotations

# Preserve caller-specific encodings/block sizes; serializers stay local.

# Shared RGIC execution: IC commit/profile provenance is verified at entry.

import importlib.util
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
_dll_handles=[]
if os.name=='nt' and hasattr(os,'add_dll_directory'):
    for _path in (r'D:\miniconda\Library\bin',r'D:\miniconda\DLLs'):
        if Path(_path).is_dir():_dll_handles.append(os.add_dll_directory(_path))
import numpy as np
from threadpoolctl import threadpool_limits

from _paths import BASE, CORE, activate
activate()
SOURCES={
    'archived_report_rgc_ic':CORE/'rgc_ic.py',
    'native_estimators':BASE.parent/'estimators_shared.py',
    'ABIDE_preparation':CORE/'prepare_raw.py',
    'original_report_reader':CORE/'report_data.py',
    'original_matched_reader':CORE/'report_data_matched.py',
}
_MODULES=None


def _load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def modules():
    global _MODULES
    if _MODULES is not None:return _MODULES
    module=_load('_local_raw_archived_rgc',SOURCES['archived_report_rgc_ic'])
    preparation=_load('_local_raw_ABIDE_preparation',SOURCES['ABIDE_preparation'])
    estimators=preparation.estimator_module()
    original=_load('_local_raw_original_report',SOURCES['original_report_reader'])
    matched=_load('_local_raw_matched_report',SOURCES['original_matched_reader'])
    _MODULES=dict(rgc=module,preparation=preparation,estimators=estimators,original=original,matched=matched)
    return _MODULES




















