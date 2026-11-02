"""Local numerical runtime initialization; never changes global installations."""
import os

_dll_handles = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    for _path in (r"D:\miniconda\Library\bin", r"D:\miniconda\DLLs"):
        if os.path.isdir(_path):
            _dll_handles.append(os.add_dll_directory(_path))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
