# RGIC User Guide

Version `0.1.0a3`. RGIC computes Reliability-Gated Information Connectivity and its intermediate quantities. It requires Python ≥ 3.10 and NumPy ≥ 1.23.

## 1. Installation and help

For GitHub installation, use the repository URL described in the README; Git-based installation requires a local Git client. From a downloaded or cloned source directory, run:
```bash
python -m pip install .
rgic --help
rgic fit --help
rgic manual
```
`--help` (or `-h`) displays command usage; `--need` and `---need` are aliases. `manual` displays this guide, and `rgic --version` displays the version. If the executable is unavailable, use `python -m rgic`.

## 2. Run the example

From the source directory:
```bash
python examples/python_example.py --out example_run
rgic inspect example_run/results.npz
```
The example generates three regional signals, including a delayed association. Eight reference recordings estimate reliability, and two target recordings are evaluated. Inputs, region labels, the fitted model, and results are saved. The output directory must be empty or not yet exist.

## 3. Prepare inputs

Use a two-dimensional array `(time_points, regions)` for one recording or a three-dimensional array `(recordings, time_points, regions)` for equally long recordings. Region count and column order must match across recordings; recording lengths may differ.

The CLI reads `.npy` and `.npz`. An `.npz` file can contain a single 2D/3D array named `X`. For variable lengths, store separate 2D arrays named `record_001`, `record_002`, etc.; they are read in name order. The Python API also accepts a list of 2D arrays.

Inputs must contain finite real values and varying regional signals, with at least three overlapping time points at every searched lag. The package does not read brain images or automatically filter, detrend, or remove time points. Correlation calculations recenter each overlapping segment.

Provide labels such as `["ROI_A", "ROI_B", "ROI_C"]` in `labels.json`, or a text file with one name per line. Passing labels during both fitting and transformation verifies region order. Without labels, the program assumes the order is unchanged.

## 4. Fit, then transform

```bash
rgic fit --input example_run/reference.npy --out example_run/cli_model.npz --roi-labels example_run/labels.json
rgic transform --input example_run/target.npy --model example_run/cli_model.npz --out example_run/cli_results.npz --roi-labels example_run/labels.json
```
`fit` estimates frequency, consistency, and reliability from reference recordings. `transform` applies that saved reliability to target recordings without updating it. Output files must use `.npz`; add `--overwrite` to replace existing files.

Defaults work without a configuration file. To customize them, use `rgic fit --input reference.npy --config examples/sample_config.json --out model.npz`; omitted fields retain their defaults. Refit after changing the lag range, gain threshold, or other configuration fields.

## 5. Python API

```python
import numpy as np
from rgic import RGIC, RGICConfig
from rgic.io import save_model, load_model, save_results

reference = np.load("example_run/reference.npy")
target = np.load("example_run/target.npy")
names = ["ROI_A", "ROI_B", "ROI_C"]
model = RGIC(RGICConfig(lags=(-1, 1))).fit(reference, roi_names=names)
results = model.transform(target, roi_names=names)
print(results[0].matrix("ic"))
print(results[0].matrix("lag"))
```
`transform` always returns a list, with one result per target recording. Attributes such as `result.ic` are upper-triangle edge vectors indexed by `result.pairs`; `result.matrix("ic")` expands them to full matrices. Use `save_model(model, path)`, `load_model(path)`, and `save_results(results, path)` for persistence.

## 6. Parameters and time units

| Parameter | Default | Meaning |
|---|---:|---|
| `lags` | `[-1,1]` | Nonzero integer sample offsets, supplied in positive/negative pairs |
| `lag_unit` | `samples` | Cost unit: `samples` or `seconds` |
| `tr` | `null` | Sampling interval; required in seconds mode |
| `lambda_` | 1 | Exponential temporal-cost discount |
| `gamma` | 0.2 | Exponential amplitude compensation |
| `delta` | 0 | Minimum correlation gain beyond zero lag |
| `cap` | 0.9 | Maximum compensated candidate magnitude |
| `quantile` | 0.95 | Normalization quantile across all edge evidence |
| `epsilon` | `1e-8` | Positive stabilizer in the normalization denominator |
| `tie_tolerance` | `1e-12` | Tolerance for tied absolute correlation peaks |

`lags` always specifies sample offsets. In seconds mode, `lags=[-2,-1,1,2]` with `tr=0.8` searches up to ±1.6 seconds and uses `C=tr*abs(lag)`; sample mode uses `C=abs(lag)`. The units of `lambda_` and `gamma` are inverse to those of `C`. A seconds-mode model assumes one common TR and does not resample inputs.

## 7. Calculation

For each edge, find the largest absolute nonzero-lag correlation `corr[x_i(t),x_j(t+lag)]`, giving `rstar` and `lag`; let `r0` be the zero-lag correlation. Positive lag means region i precedes region j.

Gain is `G=max(0,abs(rstar)-abs(r0)-delta)`. Across references, `F` is the fraction with positive gain, `L` is the fraction of those valid recordings sharing the modal exact signed offset, and `S` is the absolute mean peak-correlation sign over those same recordings. Reliability is `R=F*L*S`. If no reference has positive gain, all four are zero.

Target evidence is `A=R*G*exp(-lambda_*C)`, and the gate is `eta=clip(A/(Q(A)+epsilon),0,1)`. The quantile `Q` uses all undirected edges in the current recording, including zero-evidence edges. The candidate is `P=sign(rstar)*min(abs(rstar)*exp(gamma*C),cap)`.

The final connection is `IC=(1-eta)*r0+eta*P`, with strength `abs(IC)`. Zero gain or zero reliability retains the Pearson correlation.

## 8. Read outputs

`rgic inspect results.npz` lists array shapes and configuration metadata. In Python, use `np.load("results.npz", allow_pickle=False)`. Each matrix array has shape `(recordings, regions, regions)`, retaining the recording axis even for one recording.

| File field | Content |
|---|---|
| `ic`, `strength`, `pearson` | Signed connectivity, its magnitude, and zero-lag correlation |
| `lag_samples`, `lag_peak` | Selected sample offset and its signed correlation |
| `gain`, `cost`, `potential` | Gain, temporal cost, and compensated candidate |
| `evidence`, `eta` | Gate evidence and mixing weight |
| `F`, `L`, `S`, `reliability` | Reference-derived reliability components |
| `lag_ambiguous` | Flag for tied peaks in opposite temporal directions |
| `edge_index`, `metadata` | Upper-triangle edge indices and JSON metadata |

`ic`, `strength`, and `pearson` are symmetric with diagonal one. `lag_samples` is antisymmetric with diagonal zero; other components default to diagonal zero. Python's `matrix("lag")` corresponds to the file field `lag_samples`, and `matrix("opposite_tie")` corresponds to `lag_ambiguous`.

Ties choose the smallest absolute offset, then the negative offset. Opposite-direction ties set `lag_ambiguous` to true; the representative peak still enters the calculation. Inspect the flag when interpreting lag. Connection sign describes correlation polarity, whereas lag sign describes temporal order. If tied peaks also have opposite correlation signs, IC polarity depends on the representative peak; keep region order fixed to preserve that convention.

## 9. Troubleshooting

For an existing output, choose another path or add `--overwrite`. For mismatched labels, align the input columns first. Check signals for NaN, infinity, or constant overlapping segments. If longer offsets leave too little overlap, reduce the search range or provide longer recordings.

This version provides algorithm calculations, file handling, and command help. Example parameters illustrate usage; choose settings that match the sampling and processing of your signals.
