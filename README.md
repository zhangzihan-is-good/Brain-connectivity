# Reliability-Gated Information Connectivity (RGIC): research code

Reliability-Gated Information Connectivity (RGIC) combines zero-lag correlation, lag-related correlation gain, temporal discounting, amplitude compensation and cross-recording reliability. The four experiment folders implement simulated-signal analyses, anatomical-connectivity prediction in HCP, repeated-scan evaluation in HNU and autism classification in ABIDE.

This guide describes the code structure, algorithms, numerical environments, input interfaces and execution commands. Identify the software revision used in an analysis with `git rev-parse HEAD`.

## Repository structure

```text
.
├── README.md
├── run.py                      Common launcher
├── run_tests.py                Source/core/upstream checks
├── run_public_tests.py         Synthetic and shared-interface unit tests
├── rgic_shared.py             Experiment-facing RGIC interface
├── rgic_backend.py            Pinned IC primitives and compatibility profiles
├── rgic_profiles_abide.py     ABIDE group-gate/float32/alpha conventions
├── estimators_shared.py       Native estimators shared by HNU and ABIDE
├── io_shared.py               JSON/CSV and file hashing
├── shared_paths.py            Input-root resolution and compatibility defaults
├── requirements-demo.txt
├── requirements-ic.txt
├── examples/                  Synthetic example code
├── tests/                     Synthetic validation code
├── tools/                     Source and documentation checks
├── Simulation/                Main and supplementary simulated-signal analyses
├── HCP/                       Cached-feature anatomical prediction
├── HNU/                       Fixed-model test–retest analyses
├── ABIDE/                     Classification and selected sensitivity analyses
└── _vendor/                   Selected, hash-verified IC source and notices
```

Fixed parameter selections are configuration inputs. Running the programs creates new outputs at the requested locations.

## Installation and numerical environments

Use a separate environment for each analysis. Keep the versions specified in that experiment's requirements file for numerical comparisons.

| Purpose | Environment / installation |
| --- | --- |
| Simulation | Python 3.12, `python -m pip install -r Simulation/requirements.txt`. CPU; worker count is configurable. |
| HCP | Python 3.12, `python -m pip install -r HCP/requirements.txt`. CPU. |
| HNU | Python 3.9 and `python -m pip install -r HNU/requirements.txt`. CPU; optional exact-summation acceleration has a fallback. |
| ABIDE | Windows Python 3.9.7: NumPy 1.26.4, SciPy 1.10.1, scikit-learn 1.4.2, h5py 3.6.0 and the other dependencies in `ABIDE/requirements.txt`. Neural workflows use the specified PyTorch 2.5.1+cu121 environment. |

A minimal environment can be created with:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-demo.txt
```

For the ABIDE CUDA environment:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r ABIDE/requirements.txt
```

SVM/LR main refits use CPU. Neural model device settings are retained; the supplementary FCN workflow requires a compatible NVIDIA GPU/driver.

Use the dependency versions listed in each experiment's requirements file. The data-free example and shared-interface checks use the separate environment in `requirements-demo.txt`.

## Quick start

From this directory:

```bash
python -B examples/synthetic_demo.py --output outputs/demo
python -B run_tests.py
```

The demo generates eight reference and three target recordings, four regions and 256 time points. Use a new or empty output directory. It writes `summary.json` and `synthetic_results.npz`; there is no precomputed demo result in the repository.

Expected summary fields include:

```json
{
  "n_reference": 8,
  "n_targets": 3,
  "first_pair_lag_samples": [1, 1, 1],
  "reference_unchanged": true,
  "finite_outputs": true
}
```

Measured computation time was about 0.025 seconds, excluding interpreter startup, in the tested Python 3.12.14 / NumPy 2.3.5 environment. This is a synthetic library demonstration.

`run_tests.py` checks the selected IC source hashes, synthetic/shared-interface behavior and upstream scientific/file-roundtrip behavior. `run_public_tests.py` runs the synthetic/shared-interface subset.

## Path and output conventions

All commands below are executed from the repository root. Replace input placeholders with actual paths. Direct script commands retain the current working directory.

The optional common launcher is:

```bash
python -B run.py info
python -B run.py hcp -- --help
python -B run.py --python /path/to/interpreter hnu -- --help
```

It executes a selected runner in that experiment's directory. Supply **absolute input/output paths** when using this launcher. Arguments following `--` are forwarded unchanged.

Use `--inputs` for the HCP input directory, `--data-root` for HNU ROI files and `--data-workspace` for the ABIDE input root. These flags accept local paths. `FREEENERGY_WORKSPACE` is an optional compatibility environment variable read by `shared_paths.py` to choose default input paths; its name does not refer to another software dependency. Explicit path options can be used without setting it. Write generated outputs to a separate new directory.

## Original datasets and computation entry points

| Dataset | Official download page |
| --- | --- |
| HCP Young Adult | [HCP data releases and ConnectomeDB downloads](https://humanconnectome.org/study/hcp-young-adult/data-releases) |
| HNU1 | [CoRR HNU1 downloads](https://fcon_1000.projects.nitrc.org/indi/CoRR/html/hnu_1.html#downloads) |
| ABIDE I | [ABIDE I imaging and phenotypic downloads](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html); [ABIDE Preprocessed ROI downloads](https://preprocessed-connectomes-project.org/abide/download.html) |

The experiment runners have the following computation entry points. The download pages above identify the source datasets; the input types below describe what the analysis code reads.

| Experiment | Computation starts from | Code entry point |
| --- | --- | --- |
| Simulation | Signals generated by the simulation code from fixed seeds | [Simulation/experiment.py](Simulation/experiment.py) |
| HCP | Cached group-level edge features, structural weights and fixed edge-fold assignments | [HCP/evaluate.py](HCP/evaluate.py), `load_inputs` |
| HNU | Preprocessed regional time series and a local recording/group configuration | [HNU/runtime.py](HNU/runtime.py), `load_primitives`; [HNU/common.py](HNU/common.py), `config` |
| ABIDE | Cached connectivity primitives, cohort metadata and fixed train/validation/test identities; length/noise analyses also use preprocessed AAL time series | [ABIDE/main.py](ABIDE/main.py); [ABIDE/supplements/run_selected.py](ABIDE/supplements/run_selected.py) |

The runners operate at these entry points. `prepare_inputs.py` checks the corresponding input interfaces; it is not an MRI preprocessing program.

## 1. Simulated-signal analyses

**Implemented analyses:** main and supplementary simulated-signal evaluations of direction and strength. Inputs are generated from fixed seeds.

The fixed protocol has two acquisition profiles, three degradation settings and six relationship types (36 conditions). Each world has 24 pairs, 16 reference recordings and three target recordings. There are four development and ten test repeats per condition. The main direction cohort has 360 test worlds, and native strength uses 300 separately generated worlds.

RGIC uses reference-only reliability R=F×L×S. F is the fraction of reference recordings with positive lag-related correlation gain; L is the fraction of valid recordings sharing the most frequent signed lag; S is the absolute mean sign of the valid lag-peak correlations. Their product weights evidence supported across recordings. Reference recordings estimate reliability; separate target recordings are evaluated with that reliability held fixed.

The physical lag window extends up to 2.2 seconds, with lambda=2 and gamma=0.45 in inverse seconds. W1 uses offsets of ±1 sample; W2 uses ±1/±2 samples. The simulation W2 settings are lambda=2.25 and gamma=0.55 in sampling units. Constants, conditions and seeds are defined in `Simulation/core.py`; fixed experiments and statistics are in `Simulation/experiment.py`.

Run the main analysis:

```bash
python -B Simulation/experiment.py --workers 4 --output outputs/simulation/main/results
```

This computes development thresholds and grids, test direction/AP, native strength and component ablations, including the shared-measurement-noise setting. Tables are written under `outputs/simulation/main/results/`.

Run the supplementary analyses:

```bash
python -B Simulation/appendix_experiment.py --workers 4 --output outputs/simulation/appendix
```

These cover signal spectra/tSNR/autocorrelation, lag peaks, W2, profile-specific AP, lag error, noise/knee sensitivity, nearest-neighbor TE, development-selected classical settings, PSI standardization and reference-supported direction scores. Four-pair expensive-comparator and full-24-pair reference summaries are separate outputs.

Use the default development/test counts for the full simulation run. `--quick` shortens the supplementary run for a smoke check.

## 2. HCP anatomical-connectivity prediction

**Implemented analyses:** anatomical-support/weight prediction, controls, subgroup analyses and supplementary feature/network diagnostics.

The runner evaluates cached group-level features for 400 regions and 79,800 unordered edges, with 5,059 positive anatomical-support labels. Group summaries represent 33 participants × four recordings. W1 and lambda=gamma=1 use sampling-interval units. `HCP/evaluate.py` defines feature loading, predictor construction and fixed edge cross-validation; `HCP/supplement_diagnostics.py` implements supplementary diagnostics.

The file interface is defined in [HCP/prepare_inputs.py](HCP/prepare_inputs.py): `inputs.npz` for main features, `appendix_inputs.npz` for supplementary features and `sc_weights.npy` for structural weights. `HCP_INPUT_DIR` below means a directory satisfying that interface. The runner reads cached gates and potentials directly.

Compute using a new or empty output directory:

```bash
python -B HCP/run.py --inputs "HCP_INPUT_DIR" --output outputs/hcp --workers 3 --compute-only
```

`--section main` and `--section appendix` restrict the run. Main outputs include `fold_results.csv`, `node_repeat_results.csv`, `weight_repeat_results.csv`, `control_repeat_results.csv`, `stratified_results.csv` and `summary.json`. Supplementary outputs include feature contrasts, coefficients and network-pair metrics.

All classification models retain distance and absolute group Pearson correlation; the full RGIC model has 17 predictors. Cross-validation partitions edges, and repeated-partition SD summarizes variation across those partitions.

## 3. HNU test–retest analyses

**Implemented analyses:** local/global/high-gain strength, single-scan direction, consensus direction and supplementary support/orientation, roles, degree and participant-specificity readouts.

The runner starts from preprocessed regional time series. Its numerical interface uses finite float32 arrays of shape **295×109** and TR=2 seconds. `HNU/runtime.py` demeans each signal, applies fourth-order zero-phase Butterworth filtering at 0.01–0.1 Hz and computes connectivity primitives.

[HNU/common.py](HNU/common.py) expects a local JSON configuration at `HNU/input_config.json` for recording order, file identities and A/B group membership. [HNU/prepare_inputs.py](HNU/prepare_inputs.py) defines the configuration checks. `HNU_NPY_DIR` below is the directory of ROI arrays read by `HNU/runtime.py`.

Compute the analyses:

```bash
python -B HNU/run.py --data-root "HNU_NPY_DIR" --output outputs/hnu --workers 3
```

Use a new/empty computation directory. Every run computes primitives from the ROI inputs. Outputs include `summary.csv`, `paired.csv`, `individual_S.csv`, `direction_density.csv` and task NPZ files.

`HNU/models.json` contains the fixed configurations; `common.py` defines scopes, modes, seed and 5,000 whole-participant bootstrap resamples. The input contract assigns participants to group A (14 people) or B (15 people). A-to-B transfer estimates reference reliability from group A scans 1–5 and evaluates group B; B-to-A reverses those roles. The reported combined result gives these two transfers equal weight.

Within either evaluation group, `retest` compares scans 6–10 with one another. `temporal` compares each person's scans 6–10 with that same person's early scans 1–5. For strength, the five early rank profiles are averaged and normalized into an early personal template. Directional-consensus evaluation combines early selected arrows into a fixed template. Both modes use the same selected parameters and frozen reference reliability within a transfer.

The runner evaluates the fixed configurations in `HNU/models.json`. Additional `A_selected_all29` and `B_selected_all29` outputs apply the A-selected or B-selected model to all 29 people; use the held-out-group transfers for the primary analysis.

## 4. ABIDE classification and sensitivity analyses

**Implemented analyses:** autism classification, direction representation, component removal, demographic inputs, low-motion subsets, signal-length and measurement-noise analyses.

Classifier implementations include SVM, logistic regression, FCN, GCN, LSTM, Transformer and MADE. The main runner refits SVM, logistic regression, FCN, GCN and LSTM using the fixed configurations keyed by classifier, atlas, lag window and fold in `ABIDE/configs/main_models.json`; runner options are listed by `--help`.

ABIDE has one public launcher, `ABIDE/run.py`. Its implementation is organized as follows:

| Location | Execution role |
| --- | --- |
| `ABIDE/main.py` | Refit a selected main classifier/atlas/window/fold unit |
| `ABIDE/core/` | Read connectivity primitives and cohort metadata, compute RGIC gates and select features |
| `ABIDE/models/` | Fit and evaluate linear/neural classifiers using the bundled network implementations and defaults |
| `ABIDE/supplements/` | Execute direction, component, population and signal-length/noise analyses and summarize their fitted records |
| `ABIDE/configs/main_models.json` | Fixed main-model parameters |
| `ABIDE/configs/supplement_tasks.json` | Fixed supplementary tasks, parameter candidates and identity checks |
| `ABIDE/prepare_inputs.py`, `ABIDE/tools/` | Validate inputs and the active source/configuration set |

The analyses use CPAC `filt_global` ROI products, a fixed cohort of 1,035 participants (505 ASD, 530 controls) and fixed train/validation/test identities. AAL/EZ use 116 regions (6,670 pairs); CC200 uses 200 (19,900 pairs).

Main classification starts from cached connectivity primitives and cohort/fold metadata. [ABIDE/main.py](ABIDE/main.py) loads these inputs, selects features and refits the classifier specified in [main_models.json](ABIDE/configs/main_models.json). Selected supplementary tasks are defined in [supplement_tasks.json](ABIDE/configs/supplement_tasks.json); length/noise tasks also read preprocessed AAL ROI time series. `DATA_WORKSPACE` below is the local input root used by the ABIDE loaders. [ABIDE/prepare_inputs.py](ABIDE/prepare_inputs.py) validates the loader interface.

Read-only checks and execution plans:

```bash
python -B ABIDE/run.py --mode verify
python -B ABIDE/run.py --mode plan-main
python -B ABIDE/run.py --mode supplements --plan
python -B ABIDE/prepare_inputs.py --data-workspace "DATA_WORKSPACE" --list
python -B ABIDE/tools/check_identities.py --data-workspace "DATA_WORKSPACE"
```

Refit one fixed main unit using a new/empty output directory:

```bash
python -B ABIDE/run.py --mode main --classifier logistic --atlas aal --window 1 --fold 0 --data-workspace "DATA_WORKSPACE" --output "NEW_ABIDE_RUN"
```

The default fits the methods for that unit; `--method RGIC` selects RGIC alone. Generated outputs include the protocol, per-method predictions/scores/metrics and model files. This command refits the selected configuration for the requested unit.

Run selected supplementary experiments:

```bash
python -B ABIDE/run.py --mode supplements --execute --data-workspace "DATA_WORKSPACE" --output "NEW_ABIDE_SUPPLEMENTS" --rebuild-raw
```

`--rebuild-raw` builds the shared 13-condition connectivity-primitive cache from preprocessed AAL ROI time series in the new output directory before model-task filtering. Here “raw” names this primitive cache. `--family`, `--classifier` and `--fold` restrict the subsequent classifier fits. The complete selection has 5,720 fold records in 572 groups. The low-motion subset requires finite `0 <= func_mean_fd <= 0.2` and filters the fixed roles without a new split.

After all selected records complete, summarize the same output directory. Summaries use 500 conditional paired bootstrap resamples with seed 20261030:

```bash
python -B ABIDE/run.py --mode supplements --execute --action summarize --data-workspace "DATA_WORKSPACE" --output "NEW_ABIDE_SUPPLEMENTS"
```

**Interpretation:** `protocol.full_labels_used_for_F=true` in `main_models.json` means supervised F screening uses train, validation and test labels. The resulting AUROC describes discrimination under this screening design rather than independent held-out feature selection. Only RGIC received the broad validation search, with comparator methods using matched selected settings. Demographics enter the classifier, not the RGIC formula. Fixed feature columns still involve classifier refitting; truncation/noise experiments refit classifiers under each condition.

## AI assistance and human responsibility

Development used AI-assisted coding for implementation, testing and documentation. Human authors have reviewed the mathematical definitions, reference-state isolation, input handling, numerical behavior, experimental protocols and scientific interpretation, and remains responsible for the code, its scientific interpretation, and release decisions.
