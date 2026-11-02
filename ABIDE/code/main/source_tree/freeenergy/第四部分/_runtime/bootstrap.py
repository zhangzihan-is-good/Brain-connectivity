"""Use the existing scientific Python stack without changing installed packages.

Validated interpreter: D:\\miniconda\\envs\\temp\\python.exe (Python 3.9.0).
Import this module before numpy/scipy/h5py/matplotlib.  DLL directory handles
must remain alive for the lifetime of this process.
"""
import importlib.util
import os
import sys
from pathlib import Path

DLL_HANDLES = []


def configure():
    for directory in (r"D:\miniconda\envs\temp\Library\bin",
                      r"D:\miniconda\Library\bin"):
        if hasattr(os, "add_dll_directory") and os.path.isdir(directory):
            DLL_HANDLES.append(os.add_dll_directory(directory))
    if importlib.util.find_spec("h5py") is None:
        package = Path(r"D:\miniconda\Lib\site-packages\h5py")
        spec = importlib.util.spec_from_file_location(
            "h5py", str(package / "__init__.py"),
            submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["h5py"] = module
        spec.loader.exec_module(module)


configure()
