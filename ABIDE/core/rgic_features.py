"""ABIDE training-group gate operations through the shared RGIC interface."""
import sys as _shared_sys
from pathlib import Path as _SharedPath
_shared_root = next(p for p in _SharedPath(__file__).resolve().parents if (p / "rgic_shared.py").is_file())
if str(_shared_root) not in _shared_sys.path:
    _shared_sys.path.insert(0, str(_shared_root))
import rgic_shared as _rgic_shared


def gate_components(gain, cost, lag, rho):
    return _rgic_shared.abide_gate_components(gain, cost, lag, rho)


def make_eta(components, lambda_cost, percentile):
    return _rgic_shared.abide_eta(components, lambda_cost, percentile)

