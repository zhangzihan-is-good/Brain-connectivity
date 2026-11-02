"""Compatibility exports for the shared frozen native estimators.

All public/private functions and constants keep their original identities.
"""
from pathlib import Path as _Path
import sys as _sys
_root = next(p for p in _Path(__file__).resolve().parents if (p / "estimators_shared.py").is_file())
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
import estimators_shared as _shared
globals().update({name: value for name, value in vars(_shared).items() if not name.startswith("__")})
