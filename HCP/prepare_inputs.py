"""Validate an existing HCP research-input bundle and optionally copy it unchanged.

This does not create group features, structural weights or CV assignments from
raw imaging. Obtain the three derived files from the study authors/data holder.
Only the supplied files are copied; numerical arrays and metadata are not rewritten.
"""
from pathlib import Path
import argparse
import json
import shutil

import numpy as np
import _shared
from io_shared import sha256_file


FILES = ('inputs.npz', 'appendix_inputs.npz', 'sc_weights.npy')
FLOAT_FIELDS = (
    'distance', 'r0', 'mean_gain', 'mean_cost', 'freq', 'lag_consistency',
    'sign_consistency', 'reliability', 'gain_cost_evidence', 'raw_eta', 'eta',
    'q_gain', 'q_cost', 'q_evidence', 'potential',
)
NATIVE_FIELDS = (
    'native_lag_peak', 'native_gaussian_te', 'native_var_granger',
    'native_coherence', 'native_psi', 'native_gcmi', 'native_phiid',
)
INTEGER_FIELDS = {
    'edge_i': ((79800,), 'int64'), 'edge_j': ((79800,), 'int64'),
    'sc_support': ((79800,), 'int8'), 'cv_folds': ((20, 79800), 'int8'),
    'network_labels': ((400,), 'int64'),
}
FIXED_CONFIGURATION = {
    'W': 1, 'lambda': 1, 'gamma': 1, 'delta': 0, 'cap': 0.9,
    'time_units': 'sampling offsets',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_archive(path, main):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    fields = FLOAT_FIELDS + (NATIVE_FIELDS if main else ())
    required = set(fields) | set(INTEGER_FIELDS) | {'metadata_json'}
    require(required <= set(data), f'{path.name}: missing keys {sorted(required - set(data))}')
    for key in fields:
        value = data[key]
        require(value.shape == (79800,) and value.dtype == np.dtype('float64'),
                f'{path.name}: {key} must be float64 with shape (79800,)')
        require(np.isfinite(value).all(), f'{path.name}: {key} contains nonfinite values')
    for key, (shape, dtype) in INTEGER_FIELDS.items():
        require(data[key].shape == shape and data[key].dtype == np.dtype(dtype),
                f'{path.name}: {key} must be {dtype} with shape {shape}')
    ei, ej = np.triu_indices(400, 1)
    require(np.array_equal(data['edge_i'], ei) and np.array_equal(data['edge_j'], ej),
            f'{path.name}: edges must follow np.triu_indices(400, 1) order')
    y = data['sc_support']
    require(np.isin(y, (0, 1)).all() and int(y.sum()) == 5059,
            f'{path.name}: expected binary support with 5059 positive edges')
    for repeat, assignments in enumerate(data['cv_folds']):
        require(np.array_equal(np.unique(assignments), np.arange(5)),
                f'{path.name}: repeat {repeat} must contain fold IDs 0..4')
        for fold in range(5):
            require(np.unique(y[assignments == fold]).size == 2,
                    f'{path.name}: repeat {repeat}, fold {fold} lacks one support class')
    require(np.array_equal(np.unique(data['network_labels']), np.arange(7)),
            f'{path.name}: network_labels must contain exactly the codes 0..6')
    raw = data['metadata_json']
    require(raw.shape == () and raw.dtype.kind == 'U',
            f'{path.name}: metadata_json must be a scalar Unicode JSON string')
    metadata = json.loads(str(raw))
    require(isinstance(metadata, dict), f'{path.name}: metadata_json must encode an object')
    config = metadata.get('fixed_configuration')
    require(isinstance(config, dict), f'{path.name}: missing fixed_configuration object')
    for key, value in FIXED_CONFIGURATION.items():
        require(config.get(key) == value,
                f'{path.name}: fixed_configuration.{key} must be {value!r}')
    if main:
        require(isinstance(metadata.get('configuration_origin'), str)
                and bool(metadata['configuration_origin'].strip()),
                f'{path.name}: configuration_origin must document the earlier selection')
    return data


def validate_bundle(source):
    """Check the retained study schema without fitting, estimating or coercing arrays."""
    source = Path(source).expanduser().resolve()
    for name in FILES:
        require((source / name).is_file(), f'Missing research input: {source / name}')
    before = {name: sha256_file(source / name) for name in FILES}
    main = load_archive(source / FILES[0], True)
    appendix = load_archive(source / FILES[1], False)
    for key in FLOAT_FIELDS + tuple(INTEGER_FIELDS):
        require(main[key].dtype == appendix[key].dtype
                and np.array_equal(main[key], appendix[key]),
                f'The two archives disagree on retained array {key}')
    weights = np.load(source / FILES[2], allow_pickle=False)
    require(weights.shape == (400, 400) and weights.dtype == np.dtype('float64'),
            'sc_weights.npy must be float64 with shape (400, 400)')
    require(np.isfinite(weights).all() and np.all(weights >= 0)
            and np.array_equal(weights, weights.T) and np.all(np.diag(weights) == 0),
            'sc_weights.npy must be finite, nonnegative, symmetric and zero diagonal')
    require(np.array_equal(weights[np.triu_indices(400, 1)] > 0, main['sc_support'] == 1),
            'Structural weights do not match sc_support in upper-triangle order')
    require(before == {name: sha256_file(source / name) for name in FILES},
            'Input files changed during validation')
    return dict(status='valid', source=str(source), files_sha256=before,
                n_regions=400, n_edges=79800, n_positive=5059,
                cv_repeats=20, folds_per_repeat=5,
                scope='Retained research-input schema; no raw-imaging reconstruction or identity authentication')


def prepare(source, output=None):
    report = validate_bundle(source)
    if output is not None:
        destination = Path(output).expanduser().resolve()
        require(destination != Path(report['source']), 'Output must differ from source')
        require(not destination.exists() or (destination.is_dir() and not any(destination.iterdir())),
                'Output must be a new or empty directory; existing files are never overwritten')
        destination.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            shutil.copyfile(Path(report['source']) / name, destination / name)
            require(sha256_file(destination / name) == report['files_sha256'][name],
                    f'Byte-preserving copy verification failed: {name}')
        report.update(status='copied_and_validated', output=str(destination))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Directory containing the three author-provided derived research inputs')
    parser.add_argument('--output', type=Path,
                        help='Optional new/empty directory for byte-identical copies; omit to validate only')
    args = parser.parse_args()
    try:
        report = prepare(args.source, args.output)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f'HCP input validation failed: {error}\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
