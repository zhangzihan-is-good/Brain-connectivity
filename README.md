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
├── shared_paths.py            External workspace/data-path resolution
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

`FREEENERGY_WORKSPACE` identifies the directory containing an experiment's external data layout. The explicit `--inputs`, `--data-root` and `--data-workspace` options select the locations used by each command below. Write generated outputs to a separate new directory.

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

Provide one directory containing:

| File | Required content |
| --- | --- |
| `inputs.npz` | Group RGIC descriptors, seven native comparator vectors, upper-triangle edge identities, structural support, network labels, fixed CV assignments and `metadata_json`. |
| `appendix_inputs.npz` | The retained group features and edge/CV/network information used by supplementary diagnostics. |
| `sc_weights.npy` | Symmetric 400×400 structural weight matrix with matching region order. |

The contract uses 400 regions, 79,800 unordered upper-triangle edges, 5,059 positive structural-support labels and 20×79,800 fixed fivefold assignments. Group summaries represent 33 participants × four recordings. W1 and lambda=gamma=1 in sampling units are retained. Both NPZ files require `metadata_json.fixed_configuration`; the main input also requires `metadata_json.configuration_origin`. Supplementary network labels must be encoded exactly as integers 0–6. Cache arrays, metadata and region/edge order must match the checks in `HCP/evaluate.py` and `HCP/supplement_diagnostics.py`.

Both NPZ files store float64 edge vectors `distance`, `r0`, `mean_gain`, `mean_cost`, `freq`, `lag_consistency`, `sign_consistency`, `reliability`, `gain_cost_evidence`, `raw_eta`, `eta`, `q_gain`, `q_cost`, `q_evidence` and `potential`. Each has shape `(79800,)`. Main inputs also contain float64 vectors `native_lag_peak`, `native_gaussian_te`, `native_var_granger`, `native_coherence`, `native_psi`, `native_gcmi` and `native_phiid` with the same shape. Edge identities `edge_i`/`edge_j` are int64 vectors following `np.triu_indices(400, 1)`. `sc_support` is an int8 binary vector; `cv_folds` is int8 with shape `(20, 79800)` and fold IDs 0–4; `network_labels` is int64 with shape `(400,)` and codes 0–6. `metadata_json` is a zero-dimensional Unicode JSON string, with `fixed_configuration` containing `W=1`, `lambda=1`, `gamma=1`, `delta=0`, `cap=0.9` and `time_units="sampling offsets"`. `sc_weights.npy` is a symmetric, nonnegative, zero-diagonal float64 matrix whose nonzero upper-triangle entries match `sc_support`.

Validate the supplied folder, or copy its validated files to a new directory:

```bash
python -B HCP/prepare_inputs.py --source "HCP_BUNDLE_DIR"
python -B HCP/prepare_inputs.py --source "HCP_BUNDLE_DIR" --output "HCP_INPUT_DIR"
```

This checks the two archives, their common arrays, fixed metadata, folds and matching structural weights. The copy operation preserves the original file bytes.

Compute using a new or empty output directory:

```bash
python -B HCP/run.py --inputs "HCP_INPUT_DIR" --output outputs/hcp --workers 3 --compute-only
```

`--section main` and `--section appendix` restrict the run. Main outputs include `fold_results.csv`, `node_repeat_results.csv`, `weight_repeat_results.csv`, `control_repeat_results.csv`, `stratified_results.csv` and `summary.json`. Supplementary outputs include feature contrasts, coefficients and network-pair metrics.

This workflow consumes prepared group-level caches with fixed gates and potentials. All classification models retain distance and absolute group Pearson correlation; the full RGIC model has 17 predictors. Cross-validation partitions edges, and repeated-partition SD summarizes variation across those partitions.

## 3. HNU test–retest analyses

**Implemented analyses:** local/global/high-gain strength, single-scan direction, consensus direction and supplementary support/orientation, roles, degree and participant-specificity readouts.

The HNU input interface accepts 290 preprocessed NPY files (29 participants × ten recordings) and a JSON configuration. Its `input_contract` describes array shape/dtype/TR; `roi_order` specifies the 109 columns; `halves` records the A/B participant groups; and `recordings` records each file's participant, scan order, session, filename and SHA256.

Each NPY must be finite float32 with shape **295×109**, matching the contract's identities, ordering and hashes. These inputs have the first five volumes removed and are standardized region-wise. The runner then demeans each signal and applies fourth-order zero-phase Butterworth filtering at 0.01–0.1 Hz with TR=2 seconds.

The first command validates the supplied contract and ROI folder. For the first run in a fresh checkout, also execute the second command: `--install` copies the supplied contract byte-for-byte to the runner's local `HNU/input_config.json`.

```bash
python -B HNU/prepare_inputs.py --contract "HNU_INPUT_CONTRACT" --data-root "HNU_NPY_DIR"
python -B HNU/prepare_inputs.py --contract "HNU_INPUT_CONTRACT" --data-root "HNU_NPY_DIR" --install
```

The tool checks all 290 files and the contract's participant/scan/ROI mapping. An existing different local contract is preserved and reported as an error.

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

Classifier implementations include SVM, logistic regression, FCN, GCN, LSTM, Transformer and MADE. Fixed refits use the configurations keyed by classifier, atlas, lag window and fold in `ABIDE/configs/reproduction/`; runner options are listed by `--help`.

The ABIDE workflow uses CPAC `filt_global` ROI products and a fixed cohort of 1,035 participants (505 ASD, 530 controls).

[ABIDE/configs/main/input_contract.json](ABIDE/configs/main/input_contract.json) lists the 94 base input files and recorded array hashes. `DATA_WORKSPACE` is the input root passed to `--data-workspace`. Paths below it are literal keys in the input format:

```text
DATA_WORKSPACE/
  dataset/abide_data/01_subject_metadata/
  dataset/abide_data/03_pearson_fc/
  dataset/abide_data/04_rgc_intermediate_data/
  第四部分/inputs/
  第四部分/按论文报告复现_20261027/inputs/
```

| Literal package directory | Meaning |
| --- | --- |
| `dataset/abide_data/` | Phenotype metadata and HDF5 files containing cohort identities, roles and connectivity primitives |
| `第四部分/inputs/` | Native-estimator NPZ caches, validation metadata, raw ROI series and TR/source manifests |
| `第四部分/按论文报告复现_20261027/inputs/` | `report_base` and `native_1035` NPY arrays aligned to the 1,035-person cohort, with their recipe manifests |
| `第三部分最终版/code/estimators.py` and `第四部分/code/prepare_raw.py` | Source copies used by the native-cache provenance checks |
| `第四部分/ABIDE后续分类器队列_20261027/inputs/` | Cohort and fixed-fold metadata for signal-length/noise analyses |

The Chinese directory names are fixed on-disk keys for the input types in this table. The input reader uses those keys together with the source-reference files and manifests.

The primary files are `dataset/abide_data/04_rgc_intermediate_data/rgc_base.hdf5` and `dataset/abide_data/03_pearson_fc/abide.hdf5`. The base HDF5 uses `patients/{FILE_ID}` with `y` (ASD=0, control=1), `site`, `sex` and `age` attributes. The Pearson file stores the ordered train/validation/test identities at `experiments/{aal,cc200,ez}_whole/{0..9}/{train,valid,test}`. The phenotype CSV is `dataset/abide_data/01_subject_metadata/Phenotypic_V1_0b_preprocessed1.csv`, with `FILE_ID`, `DX_GROUP`, `SEX`, `AGE_AT_SCAN`, `HANDEDNESS_CATEGORY`, `FIQ` and `func_mean_fd` fields.

HDF5/metadata bind participant identities, labels, atlas-specific train/validation/test roles and ROI order. AAL/EZ use 116 regions (6,670 pairs); CC200 uses 200 (19,900 pairs). The checker accepts a package when its NPZ/validation JSON, aligned NPY arrays, metadata/source recipes, runtime versions and identity hashes match the fixed contracts.

The native-cache reader hashes the two source-reference files named in the table. Identical source bytes are included as `estimators_shared.py` and `ABIDE/code/main/source_tree/freeenergy/第四部分/code/prepare_raw.py`. A recipe manifest records source hashes, runtime versions, array identities and absolute source-path keys. The checker reports the manifest's declared root, the current resolved workspace root and whether the path keys match. Source-path keys are compared literally after resolving the workspace.

Signal-length/noise analyses additionally require `第四部分/ABIDE后续分类器队列_20261027/inputs/original_cohort_and_folds.json` and the AAL recordings at `第四部分/inputs/raw_timeseries/aal/{FILE_ID}_rois_aal.1D`, including the matching ROI/header/TR metadata specified by the input contract.

Check inputs before fitting:

```bash
python -B ABIDE/prepare_inputs.py --data-workspace "DATA_WORKSPACE" --list
python -B ABIDE/prepare_inputs.py --data-workspace "DATA_WORKSPACE" --hashes --identities
python -B ABIDE/prepare_inputs.py --data-workspace "DATA_WORKSPACE" --scope supplements --hashes --identities
```

The default check validates paths and recipe metadata. `--list` prints the exact relative file keys and roles. `--hashes` also reads the large data files and verifies their recorded SHA256 values; `--identities` checks cohort/fold/low-motion identities using the ABIDE environment. The tool reports the check level and creates no links, copies, splits or fitted models.

Read-only checks and execution plans:

```bash
python -B ABIDE/run.py --mode verify
python -B ABIDE/run.py --mode plan-main
python -B ABIDE/run.py --mode supplements --plan
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

`--family`, `--classifier` and `--fold` restrict model tasks. The complete selection has 5,720 fold records in 572 groups. The task configuration retains fixed conditions and identity-hash checks; the runner derives participant axes from the package's HDF5 and phenotype metadata. The low-motion subset requires finite `0 <= func_mean_fd <= 0.2` and filters the original roles without a new split. `--rebuild-raw` builds declarations/inputs in the new output with source and recipe bindings.

After all selected records complete, summarize the same output directory. Summaries use 500 conditional paired bootstrap resamples with seed 20261030:

```bash
python -B ABIDE/run.py --mode supplements --execute --action summarize --data-workspace "DATA_WORKSPACE" --output "NEW_ABIDE_SUPPLEMENTS"
```

**Interpretation:** supervised F screening uses train, validation and test labels. This reproduces the original protocol; the resulting AUROC is not an independent leakage-free generalization estimate. Only RGIC received the broad validation search, with comparator methods using matched selected settings. Demographics enter the classifier, not the RGIC formula. Fixed feature columns still involve classifier refitting; truncation/noise experiments do not mean applying a clean-trained model unchanged.

## Shared RGIC implementation and checks

IC version `0.1.0a3` is pinned at [58625fc4a3afd69d451f27b5c279a3c2a3cae90f](https://github.com/zhangzihan-is-good/IC/commit/58625fc4a3afd69d451f27b5c279a3c2a3cae90f). `_vendor/IC_PROVENANCE.json` records the selected unmodified runtime source, required CLI resource, notice and synthetic upstream checks.

The required CLI manual resource retains its upstream text and hash. Installation and example paths in that resource refer to the complete upstream IC checkout; use this root README's commands for the selected source bundled here.

The shared adapter calls the pinned `_gate` and `_potential` primitives. A numerical profile specifies arithmetic precision, operation order, normalization scope and directional readout for an experiment. Direct compensation profiles use exp/multiply arithmetic, while the raw public API uses log-space arithmetic. HCP consumes cached group gates/potentials and shares their fusion. ABIDE uses its training-group gate, selected normalization percentile and alpha scaling. Each experiment selects its numerical profile through the shared interface.

HNU and ABIDE share the same native estimator implementation; project-specific edge order, lag/tie policies, dtype, reference selection, group aggregation and directional readouts remain explicit. The upstream `rgic` namespace is isolated from HNU's local `rgic.py`.

Run `python -B run_tests.py` for the checks included in this checkout: pinned upstream source hashes, synthetic reference/target separation, known lag direction, input/order validation, numerical profiles and file round trips. These checks generate their inputs locally. Lag direction characterizes observed BOLD temporal ordering; native strength errors compare estimators with different scales.

For code issues, include the command, environment, a minimal synthetic reproducer and expected/actual behavior.

## AI assistance and human responsibility

Development used AI-assisted coding for implementation, testing and documentation. Human authors are responsible for reviewing the mathematical definitions, reference-state isolation, input handling, numerical behavior, experimental protocols and scientific interpretation, and for correctness and release decisions.
