"""Safe NPY/NPZ and JSON I/O for the RGIC prototype (no pickle)."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from .core import RGIC, RGICConfig


PACKAGE_VERSION = "0.1.0a3"
SCHEMA_VERSION = 2


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_recordings(path):
    """Load (T,R)/(N,T,R) NPY, or X/record_* arrays from an NPZ."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        records = np.load(path, allow_pickle=False)
        if records.ndim not in (2, 3):
            raise ValueError("NPY input must have shape (T,R) or (N,T,R), with time before regions.")
        return records
    if path.suffix.lower() != ".npz":
        raise ValueError("Input must be .npy or .npz; convert ROI time series to one of these formats first.")
    with np.load(path, allow_pickle=False) as archive:
        if "X" in archive:
            if any(key.startswith("record_") for key in archive.files):
                raise ValueError("NPZ cannot contain both X and record_*; use one record format.")
            value = archive["X"]
            if value.ndim not in (2, 3):
                raise ValueError("NPZ array X must have shape (T,R) or (N,T,R).")
            return value.copy()
        keys = sorted(key for key in archive.files if key.startswith("record_"))
        if not keys:
            raise ValueError("NPZ input must contain X or 2D records named record_000, record_001, etc.")
        records = [archive[key].copy() for key in keys]
        if any(value.ndim != 2 for value in records):
            raise ValueError("Every record_* array must have shape (T,R).")
        return records


def load_roi_names(path):
    """Read a JSON list or one ROI label per line in UTF-8 text."""
    if path is None:
        return None
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    values = json.loads(text) if path.suffix.lower() == ".json" else text.splitlines()
    if not isinstance(values, list) or not values or any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError("ROI labels must be a nonempty list of strings or one nonempty label per line.")
    values = [value.strip() for value in values]
    if len(set(values)) != len(values):
        raise ValueError("ROI labels must be unique and follow the time-series column order.")
    return tuple(values)


def load_config(path):
    """Read exactly the public RGICConfig fields from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Configuration JSON must be an object.")
    if "lambda" in data:
        if "lambda_" in data:
            raise ValueError("Configuration cannot contain both lambda and lambda_.")
        data["lambda_"] = data.pop("lambda")
    try:
        return RGICConfig(**data)
    except TypeError as exc:
        raise ValueError(f"Invalid configuration fields: {exc}") from exc


def _metadata(archive):
    if "metadata" not in archive:
        raise ValueError("File is missing RGIC metadata.")
    value = archive["metadata"]
    if value.shape != () or value.dtype.kind not in "US":
        raise ValueError("metadata must be a JSON string scalar.")
    try:
        result = json.loads(value.item())
    except (ValueError, TypeError) as exc:
        raise ValueError("metadata is not valid JSON.") from exc
    if not isinstance(result, dict):
        raise ValueError("metadata must contain a JSON object.")
    return result


def _write_npz(path, arrays, metadata, *, overwrite=False):
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Model and result output filenames must end in .npz.")
    if path.exists() and not overwrite:
        raise ValueError(f"Output already exists: {path}; use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload["metadata"] = np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".rgic-", suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **payload)
        if path.exists() and not overwrite:
            raise ValueError(f"Output already exists: {path}; use --overwrite to replace it.")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def save_model(model, path, *, source_path=None, overwrite=False):
    """Save fitted reference state without Python object serialization."""
    if not hasattr(model, "reliability_"):
        raise ValueError("Model must be fitted before its reference state can be saved.")
    if model.config != model.config_:
        raise ValueError("Configuration changed after fit; refit before saving.")
    metadata = {
        "format": "rgic-model", "schema_version": SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(model.config_), "n_regions": model.n_regions_,
        "n_reference": model.n_reference_,
        "roi_names": list(model.roi_names_) if model.roi_names_ is not None else None,
        "dtype": "float64", "reliability_source": "explicit_reference_records",
    }
    if source_path is not None:
        metadata["reference_file_sha256"] = sha256_file(source_path)
    arrays = {name: getattr(model, name + "_") for name in ("F", "L", "S", "reliability")}
    return _write_npz(path, arrays, metadata, overwrite=overwrite)


def load_model(path):
    """Restore and validate a model created by save_model."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = _metadata(archive)
        if metadata.get("format") != "rgic-model" or metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported RGIC model format.")
        n_regions = metadata.get("n_regions")
        n_reference = metadata.get("n_reference")
        if type(n_regions) is not int or n_regions < 2 or type(n_reference) is not int or n_reference < 1:
            raise ValueError("Model region count or reference record count is invalid.")
        config = RGICConfig(**metadata["config"])
        size = n_regions * (n_regions - 1) // 2
        arrays = {}
        for name in ("F", "L", "S", "reliability"):
            if name not in archive:
                raise ValueError(f"Model is missing array {name}.")
            raw = archive[name]
            if raw.dtype.kind not in "fiu":
                raise ValueError(f"Model array {name} must contain real numbers.")
            value = np.asarray(raw, dtype=np.float64)
            if value.shape != (size,) or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
                raise ValueError(f"Model array {name} has an invalid shape or values.")
            arrays[name] = value.copy()
        if not np.allclose(arrays["F"] * arrays["L"] * arrays["S"], arrays["reliability"], rtol=1e-10, atol=1e-12):
            raise ValueError("Model reliability does not equal F*L*S.")
    labels = metadata.get("roi_names")
    if labels is not None:
        if not isinstance(labels, list) or len(labels) != n_regions or any(not isinstance(x, str) or not x for x in labels) or len(set(labels)) != len(labels):
            raise ValueError("Model ROI labels are invalid.")
        labels = tuple(labels)
    model = RGIC(config)
    model.config_ = config
    model.n_regions_ = n_regions
    model.n_reference_ = n_reference
    model.roi_names_ = labels
    for name, value in arrays.items():
        value.flags.writeable = False
        setattr(model, name + "_", value)
    return model


def save_results(results, path, *, source_path=None, model_path=None, overwrite=False):
    """Save an N×R×R matrix per output, preserving record order."""
    if not isinstance(results, (list, tuple)):
        results = [results]
    if not results:
        raise ValueError("No results to save.")
    first = results[0]
    if any(x.n_regions != first.n_regions for x in results):
        raise ValueError("Results have inconsistent region counts.")
    field_map = {
        "ic": "ic", "strength": "strength", "pearson": "r0",
        "lag_peak": "rstar", "lag_samples": "lag",
        "gain": "gain", "cost": "cost", "potential": "potential", "evidence": "evidence",
        "eta": "eta", "F": "F", "L": "L", "S": "S", "reliability": "reliability",
    }
    arrays = {output: np.stack([result.matrix(field) for result in results]) for output, field in field_map.items()}
    arrays["lag_ambiguous"] = np.stack([result.matrix("opposite_tie").astype(bool) for result in results])
    arrays["edge_index"] = np.asarray(first.pairs, dtype=np.int64)
    metadata = {
        "format": "rgic-results", "schema_version": SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_records": len(results), "n_regions": first.n_regions,
        "matrix_axis_order": ["record", "row_region", "column_region"],
        "lag_sign": "positive upper-triangle lag: row region precedes column region",
        "record_metadata": [result.metadata for result in results],
    }
    if source_path is not None:
        metadata["input_file_sha256"] = sha256_file(source_path)
    if model_path is not None:
        metadata["reference_model_sha256"] = sha256_file(model_path)
    return _write_npz(path, arrays, metadata, overwrite=overwrite)


def inspect_file(path):
    """Return JSON-safe metadata and array shapes without using pickle."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
        return {"file": path.name, "format": "npy-input", "shape": list(value.shape), "dtype": str(value.dtype)}
    if path.suffix.lower() != ".npz":
        raise ValueError("inspect only supports .npy and .npz files.")
    with np.load(path, allow_pickle=False) as archive:
        result = {"file": path.name, "arrays": {key: {"shape": list(archive[key].shape), "dtype": str(archive[key].dtype)} for key in archive.files if key != "metadata"}}
        if "metadata" in archive:
            result["metadata"] = _metadata(archive)
        return result
