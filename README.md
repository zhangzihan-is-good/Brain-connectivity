# Reliability-Gated Information Connectivity (RGIC): research code

RGIC combines zero-lag correlation, lag-related correlation gain, temporal discounting, amplitude compensation and cross-recording reliability. This repository implements four analyses: simulated signals, HCP anatomical-connectivity prediction, HNU test–retest evaluation and ABIDE autism classification.

## Code structure

| Location | Purpose |
| --- | --- |
| `Simulation/`, `HCP/`, `HNU/`, `ABIDE/` | Experiment code and fixed configurations |
| `rgic_shared.py`, `rgic_backend.py`, `rgic_profiles_abide.py` | Shared RGIC operations and experiment-specific numerical conventions |
| `estimators_shared.py` | Native connectivity estimators shared by HNU and ABIDE |
| `io_shared.py`, `shared_paths.py` | File interfaces and input-path defaults |
| `run.py` | Common experiment launcher |
| `examples/`, `tests/`, `run_tests.py` | Synthetic example and numerical checks |
| `_vendor/` | Pinned IC runtime source |

## Installation and quick start

Use a separate environment for each experiment and retain the versions in its requirements file.

| Experiment | Environment | Install |
| --- | --- | --- |
| Simulation | Python 3.12; CPU | `python -m pip install -r Simulation/requirements.txt` |
| HCP | Python 3.12; CPU | `python -m pip install -r HCP/requirements.txt` |
| HNU | Python 3.9; CPU | `python -m pip install -r HNU/requirements.txt` |
| ABIDE | Windows Python 3.9.7; neural workflows use PyTorch 2.5.1+cu121 | `python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r ABIDE/requirements.txt` |

ABIDE SVM/LR use CPU; supplementary FCN requires a compatible NVIDIA GPU/driver.

For the synthetic example and shared-interface checks, use Python 3.12:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-demo.txt
python -B examples/synthetic_demo.py --output outputs/demo
python -B run_tests.py
```

The example generates its own signals. `run_tests.py` checks pinned source hashes, numerical behavior, reference/target separation and file round trips.

## Paths and source datasets

Run commands from the repository root, replace input placeholders with local paths and use new or empty output directories. Explicit path options work without setting `FREEENERGY_WORKSPACE`, the compatibility variable for default input roots.

The optional common launcher forwards arguments after `--`. Use absolute input/output paths with this launcher:

```bash
python -B run.py info
python -B run.py hcp -- --help
```

| Dataset | Official download page |
| --- | --- |
| HCP Young Adult | [HCP data releases / ConnectomeDB](https://humanconnectome.org/study/hcp-young-adult/data-releases) |
| HNU1 | [CoRR HNU1](https://fcon_1000.projects.nitrc.org/indi/CoRR/html/hnu_1.html#downloads) |
| ABIDE I | [Imaging and phenotypic data](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html); [preprocessed ROI time series](https://preprocessed-connectomes-project.org/abide/download.html) |

The runners start from the input types described below. The `prepare_inputs.py` utilities validate those interfaces; MRI preprocessing precedes these analysis entry points.

## 1. Simulated-signal analyses

Inputs are generated from fixed seeds. [Simulation/core.py](Simulation/core.py) defines the signal conditions and numerical settings. Reference recordings estimate reliability `R=F*L*S`: positive-gain frequency, signed-lag consistency and correlation-sign consistency. Separate target recordings use that fixed reliability.

```bash
python -B Simulation/experiment.py --workers 4 --output outputs/simulation/main/results
python -B Simulation/appendix_experiment.py --workers 4 --output outputs/simulation/appendix
```

Use the default repeat counts for the full analyses. Supplementary `--quick` runs a shorter smoke check.

## 2. HCP anatomical-connectivity prediction

The runner starts from cached group-level edge features, structural weights and fixed edge-fold assignments for 400 regions. Its input directory contains `inputs.npz`, `appendix_inputs.npz` and `sc_weights.npy`; [HCP/prepare_inputs.py](HCP/prepare_inputs.py) defines the file interface.

```bash
python -B HCP/run.py --inputs "HCP_INPUT_DIR" --output outputs/hcp --workers 3 --compute-only
```

`--section main` or `--section appendix` limits the run. [HCP/evaluate.py](HCP/evaluate.py) implements anatomical-support/weight prediction; [HCP/supplement_diagnostics.py](HCP/supplement_diagnostics.py) implements supplementary diagnostics. The runner uses cached gates and potentials, with W1 and lambda=gamma=1 in sampling units. Cross-validation partitions edges; all classification models retain distance and absolute group Pearson correlation.

## 3. HNU test–retest evaluation

The runner starts from finite float32 ROI arrays of shape **295×109**, with TR=2 seconds. [HNU/common.py](HNU/common.py) reads a local `HNU/input_config.json` for recording identities/order and A/B membership; [HNU/prepare_inputs.py](HNU/prepare_inputs.py) defines the configuration checks. [HNU/runtime.py](HNU/runtime.py) demeans, filters at 0.01–0.1 Hz and computes connectivity primitives.

```bash
python -B HNU/run.py --data-root "HNU_NPY_DIR" --output outputs/hnu --workers 3
```

[HNU/models.json](HNU/models.json) supplies fixed parameters. A-to-B and B-to-A transfers hold reference reliability fixed during evaluation. `retest` compares scans 6–10 with one another; `temporal` compares them with personal templates from scans 1–5.

## 4. ABIDE classification and sensitivity analyses

`ABIDE/run.py` is the public launcher. `core/` contains input readers and feature selection, `models/` contains classifiers, and `supplements/` implements direction, component, population and signal-length/noise analyses.

Main classification starts from cached connectivity primitives, cohort metadata and fixed train/validation/test identities for 1,035 participants. Length/noise analyses also use preprocessed AAL ROI series. The analyses use CPAC `filt_global` products with AAL, CC200 and EZ atlases.

Fixed settings are stored in [main_models.json](ABIDE/configs/main_models.json) and [supplement_tasks.json](ABIDE/configs/supplement_tasks.json). Check inputs and inspect the selected execution plan:

```bash
python -B ABIDE/prepare_inputs.py --data-workspace "DATA_WORKSPACE" --list
python -B ABIDE/run.py --mode plan-main
python -B ABIDE/run.py --mode supplements --plan
```

Run one main classifier/atlas/window/fold unit:

```bash
python -B ABIDE/run.py --mode main --classifier logistic --atlas aal --window 1 --fold 0 --data-workspace "DATA_WORKSPACE" --output "NEW_ABIDE_RUN"
```

The default runs all methods for that unit; add `--method RGIC` for RGIC alone.

Run and summarize the fixed supplementary tasks:

```bash
python -B ABIDE/run.py --mode supplements --execute --data-workspace "DATA_WORKSPACE" --output "NEW_ABIDE_SUPPLEMENTS" --rebuild-raw
python -B ABIDE/run.py --mode supplements --execute --action summarize --data-workspace "DATA_WORKSPACE" --output "NEW_ABIDE_SUPPLEMENTS"
```

`--rebuild-raw` builds the 13-condition connectivity-primitive cache from preprocessed AAL series before task filtering. `--family`, `--classifier` and `--fold` limit subsequent classifier fits. Summarize after all tasks in `supplement_tasks.json` are complete.

## AI assistance and human responsibility

Development used AI-assisted coding for implementation, testing and documentation. Human authors have reviewed the code and scientific methods, and remain responsible for correctness, scientific interpretation and release decisions.
