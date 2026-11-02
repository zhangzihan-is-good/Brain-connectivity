# Reliability-Gated Information Connectivity (RGIC): research code

Code for the simulation, HCP, HNU and ABIDE analyses accompanying the RGIC manuscript. This repository contains scientific code, dependencies, fixed parameters, input contracts, source notices and synthetic checks.

## Repository structure

```text
.
├── README.md
├── LICENSE                    MIT license
├── run.py                      Common launcher
├── run_tests.py                Dataset-free source/core/upstream checks
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
├── tests/                     Data-free validation code; no result fixtures
├── tools/                     Source/documentation/release checking code
├── Simulation/                Main and supplementary simulated-signal analyses
├── HCP/                       Cached-feature anatomical prediction
├── HNU/                       Fixed-model test–retest analyses
├── ABIDE/                     Classification and selected sensitivity analyses
└── _vendor/                   Selected, hash-verified IC source and notices
```

Necessary final parameter selections are configuration inputs, not precomputed performance tables. Human-study measurements and derived input bundles must be obtained separately. Running the programs creates new outputs at the requested locations.

## Installation and numerical environments

Use a separate environment for each analysis. Combining the requirements files or upgrading a historical environment can change numerical results.

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

For the archived ABIDE CUDA installation target:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r ABIDE/requirements.txt
```

SVM/LR main refits use CPU. Neural model device settings are retained; the supplementary FCN workflow requires a compatible NVIDIA GPU/driver.

Actual numerical comparisons used existing Windows environments. Historical simulation/HCP regression used Python 3.9.7, NumPy 1.21.5, SciPy 1.10.1, pandas 1.3.0 and scikit-learn 1.4.2. HNU used its historical environment. ABIDE comparisons included its NumPy 1.26.4 environment and a separate NumPy 2.3.5 environment. Old/new code was compared within the same environment.

Keep the specified versions when reproducing a recorded numerical comparison.

## Quick start: no participant data required

From this directory:

```bash
python -B examples/synthetic_demo.py --output outputs/demo
python -B run_tests.py
```

The demo generates eight reference and three target recordings, four regions and 256 time points. Use a new or empty output directory. It writes `summary.json` and `synthetic_results.npz`; there is no precomputed demo result in the repository.

Expected summary fields include:

```json
{
  "participant_data_used": false,
  "n_reference": 8,
  "n_targets": 3,
  "first_pair_lag_samples": [1, 1, 1],
  "reference_unchanged": true,
  "finite_outputs": true
}
```

Measured computation time was about 0.025 seconds, excluding interpreter startup, in the tested Python 3.12.14 / NumPy 2.3.5 environment. This is a synthetic library demonstration.

`run_tests.py` checks the selected IC source hashes, the synthetic/shared-interface tests and upstream scientific/file-roundtrip tests. It uses no original sibling checkout, participant data, saved regression fixture or GPU. `run_public_tests.py` runs only the synthetic/shared-interface subset.

## Path and output conventions

All commands below are executed from the repository root. Replace input placeholders with actual paths. Direct script commands retain the current working directory.

The optional common launcher is:

```bash
python -B run.py info
python -B run.py hcp -- --help
python -B run.py --python /path/to/interpreter hnu -- --help
```

It executes a selected runner in that experiment's directory. Supply **absolute input/output paths** when using this launcher. Arguments following `--` are forwarded unchanged.

`FREEENERGY_WORKSPACE` can identify the external workspace containing `dataset/`; explicit `--inputs`, `--data-root` and `--data-workspace` parameters are preferable for independent installations. Do not write new results over original data, source, caches or historical runs.

## 1. Simulated-signal analyses

**Paper correspondence:** main and supplementary simulated-signal analyses of direction and strength. Inputs are generated from fixed seeds; no external dataset is required.

The fixed protocol has two acquisition profiles, three degradation settings and six relationship types (36 conditions). Each world has 24 pairs, 16 reference recordings and three target recordings. There are four development and ten test repeats per condition. The main direction cohort has 360 test worlds, and native strength uses 300 separately generated worlds.

RGIC uses reference-only F×L×S, a physical lag window up to 2.2 seconds, lambda=2 and gamma=0.45 in inverse seconds. W2 uses ±1/±2 sampling offsets with lambda=2.25 and gamma=0.55. Constants, conditions and seeds are defined in `Simulation/core.py`; fixed experiments and statistics are in `Simulation/experiment.py`.

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

Use the default development/test counts for the article protocol. `--quick` shortens the supplementary run for a smoke check. The main calculation took approximately 4.6–4.8 minutes in local four-thread audits using the historical environment.

## 2. HCP anatomical-connectivity prediction

**Paper correspondence:** anatomical-support/weight prediction, controls, subgroup analyses and supplementary feature/network diagnostics.

Source imaging is available through [HCP Young Adult](https://www.humanconnectome.org/study/hcp-young-adult); accept the relevant data-use terms. Structural-connectome context follows [Liu et al., NeuroImage 2023](https://doi.org/10.1016/j.neuroimage.2023.120276). The HCP workflow takes the prepared derived inputs specified below.

Provide one directory containing:

| File | Required content |
| --- | --- |
| `inputs.npz` | Group RGIC descriptors, seven native comparator vectors, upper-triangle edge identities, structural support, network labels, fixed CV assignments and `metadata_json`. |
| `appendix_inputs.npz` | The retained group features and edge/CV/network information used by supplementary diagnostics. |
| `sc_weights.npy` | Symmetric 400×400 structural weight matrix with matching region order. |

The contract uses 400 regions, 79,800 unordered upper-triangle edges, 5,059 positive structural-support labels and 20×79,800 fixed fivefold assignments. Group summaries represent 33 participants × four recordings. W1 and lambda=gamma=1 in sampling units are retained. Both NPZ files require `metadata_json.fixed_configuration`; the main input also requires `metadata_json.configuration_origin`. Supplementary network labels must be encoded exactly as integers 0–6. Cache arrays, metadata and region/edge order must match the checks in `HCP/evaluate.py` and `HCP/supplement_diagnostics.py`.

Compute using a new or empty output directory:

```bash
python -B HCP/run.py --inputs "HCP_INPUT_DIR" --output outputs/hcp --workers 3 --compute-only
```

`--section main` and `--section appendix` restrict the run. Main outputs include `fold_results.csv`, `node_repeat_results.csv`, `weight_repeat_results.csv`, `control_repeat_results.csv`, `stratified_results.csv` and `summary.json`. Supplementary outputs include feature contrasts, coefficients and network-pair metrics.

This workflow consumes prepared group-level caches with fixed gates and potentials. All classification models retain distance and absolute group Pearson correlation; the full RGIC model has 17 predictors. Cross-validation partitions edges, and repeated-partition SD summarizes variation across those partitions.

## 3. HNU test–retest analyses

**Paper correspondence:** local/global/high-gain strength, single-scan direction, consensus direction and supplementary support/orientation, roles, degree and participant-specificity readouts.

Obtain source data through [HNU1 / CoRR](https://fcon_1000.projects.nitrc.org/indi/CoRR/html/hnu_1.html), [dataset DOI](https://doi.org/10.15387/fcp_indi.corr.hnu1). Follow the provider's registration and data terms.

Place the input bundle's participant/session contract at `HNU/input_config.json` and provide its 290 preprocessed NPY files: 29 participants × ten recordings. Each file must be finite float32 with shape **295×109**, matching filenames, participant/session identities, A/B assignments, region order and SHA256 values. The supplied series have the first five volumes removed and were standardized region-wise. The runner applies the original demeaned 0.01–0.1 Hz fourth-order zero-phase Butterworth filtering with TR=2 seconds.

Compute the analyses:

```bash
python -B HNU/run.py --data-root "HNU_NPY_DIR" --output outputs/hnu --workers 3
```

Use a new/empty computation directory. Every run computes primitives from the ROI inputs. Outputs include `summary.csv`, `paired.csv`, `individual_S.csv`, `direction_density.csv` and task NPZ files.

`HNU/models.json` contains the fixed configurations; `common.py` defines scopes, modes, seed and 5,000 whole-participant bootstrap resamples. Reference reliability uses training-group scans 1–5 and is frozen for the opposite group. The two primary transfers are evaluated separately and averaged equally. Later-only and early-template comparisons share the selected model within a transfer.

The runner evaluates the fixed configurations in `HNU/models.json` for the two held-out-group transfers. All-participant selector views are descriptive analyses.

## 4. ABIDE classification and sensitivity analyses

**Paper correspondence:** autism classification, direction representation, component removal, demographic inputs, low-motion data, signal-length and measurement-noise analyses.

Classifier implementations include SVM, logistic regression, FCN, GCN, LSTM, Transformer and MADE. Fixed refits use the configurations keyed by classifier, atlas, lag window and fold in `ABIDE/configs/reproduction/`; runner options are listed by `--help`.

Obtain ABIDE-I source data from [ABIDE](https://fcon_1000.projects.nitrc.org/indi/abide/) and [preprocessed access](https://fcon_1000.projects.nitrc.org/indi/abide/preprocessed.html), following the provider's terms. This protocol uses CPAC `filt_global` ROI products and a fixed cohort of 1,035 participants (505 ASD, 530 controls).

The input contract in `ABIDE/configs/main/input_contract.json` identifies the external dependencies. Use its workspace-relative layout and required hashes; historical machine-path fields are provenance, not installation instructions. Required input families include:

```text
DATA_WORKSPACE/
  dataset/abide_data/01_subject_metadata/
  dataset/abide_data/03_pearson_fc/
  dataset/abide_data/04_rgc_intermediate_data/
  第四部分/inputs/
  第四部分/按论文报告复现_20261027/inputs/
```

The primary files are `dataset/abide_data/04_rgc_intermediate_data/rgc_base.hdf5` and `dataset/abide_data/03_pearson_fc/abide.hdf5`. The base HDF5 uses `patients/{FILE_ID}` with `y` (ASD=0, control=1), `site`, `sex` and `age` attributes. The Pearson file stores the ordered train/validation/test identities at `experiments/{aal,cc200,ez}_whole/{0..9}/{train,valid,test}`. The phenotype CSV is `dataset/abide_data/01_subject_metadata/Phenotypic_V1_0b_preprocessed1.csv`, with `FILE_ID`, `DX_GROUP`, `SEX`, `AGE_AT_SCAN`, `HANDEDNESS_CATEGORY`, `FIQ` and `func_mean_fd` fields.

HDF5/metadata bind participant identities, labels, atlas-specific train/validation/test roles and ROI order. AAL/EZ use 116 regions (6,670 pairs); CC200 uses 200 (19,900 pairs). The approved derived inputs include the original NPZ/validated JSON under `第四部分/inputs`, and the `report_base` and `native_1035` atlas-specific NPY/manifest bundles under `第四部分/按论文报告复现_20261027/inputs`. Derived arrays, metadata/source recipes, runtime and identity hashes must agree.

The native-cache reader also hashes `DATA_WORKSPACE/第三部分最终版/code/estimators.py` and `DATA_WORKSPACE/第四部分/code/prepare_raw.py`. Identical source bytes are included as `estimators_shared.py` and `ABIDE/code/main/source_tree/freeenergy/第四部分/code/prepare_raw.py`. Preserve the external input bundle's associated source copies and recipe bindings. Recipe checks require matching source bytes, runtime versions and stored absolute path strings.

Raw sensitivity runs additionally require `第四部分/ABIDE后续分类器队列_20261027/inputs/original_cohort_and_folds.json` and the approved AAL recordings at `第四部分/inputs/raw_timeseries/aal/{FILE_ID}_rois_aal.1D`, including the matching ROI/header/TR metadata specified by the input contract.

Read-only checks and execution plans:

```bash
python -B ABIDE/run.py --mode verify
python -B ABIDE/run.py --mode plan-main
python -B ABIDE/run.py --mode supplements --plan
python -B ABIDE/tools/check_identities.py --data-workspace "DATA_WORKSPACE"
```

Refit one fixed main unit using a new/empty output directory outside the code and protected data locations:

```bash
python -B ABIDE/run.py --mode main --classifier logistic --atlas aal --window 1 --fold 0 --data-workspace "DATA_WORKSPACE" --output "NEW_ABIDE_RUN"
```

The default fits the methods for that unit; `--method RGIC` selects RGIC alone. Generated outputs include the protocol, per-method predictions/scores/metrics and model files. This command refits the selected configuration for the requested unit.

Run selected supplementary experiments:

```bash
python -B ABIDE/run.py --mode supplements --execute --data-workspace "DATA_WORKSPACE" --output "NEW_ABIDE_SUPPLEMENTS" --rebuild-raw
```

`--family`, `--classifier` and `--fold` restrict model tasks. The complete selection has 5,720 fold records in 572 groups. The task configuration retains fixed conditions and identity-hash checks; the runner derives participant axes from the approved external metadata rather than shipping participant labels. The low-motion subset requires finite `0 <= func_mean_fd <= 0.2` and filters the original roles without a new split. Existing raw caches bound to different source recipes must not be silently reused. `--rebuild-raw` builds declarations/inputs in the new output.

After all selected records complete, summarize the same output directory. Summaries use 500 conditional paired bootstrap resamples with seed 20261030:

```bash
python -B ABIDE/run.py --mode supplements --execute --action summarize --data-workspace "DATA_WORKSPACE" --output "NEW_ABIDE_SUPPLEMENTS"
```

**Interpretation:** supervised F screening uses train, validation and test labels. This reproduces the original protocol; the resulting AUROC is not an independent leakage-free generalization estimate. Only RGIC received the broad validation search, with comparator methods using matched selected settings. Demographics enter the classifier, not the RGIC formula. Fixed feature columns still involve classifier refitting; truncation/noise experiments do not mean applying a clean-trained model unchanged.

## Shared RGIC implementation and validation

IC version `0.1.0a3` is pinned at [58625fc4a3afd69d451f27b5c279a3c2a3cae90f](https://github.com/zhangzihan-is-good/IC/commit/58625fc4a3afd69d451f27b5c279a3c2a3cae90f). `_vendor/IC_PROVENANCE.json` records the selected unmodified runtime source, required CLI resource, notice and synthetic upstream checks.

The required CLI manual resource retains its upstream text and hash. Installation and example paths in that resource refer to the complete upstream IC checkout; use this root README's commands for the selected source bundled here.

The adapter calls the pinned private `_gate` and `_potential` primitives while preserving experimental conventions. Direct compensation profiles retain the historical exp/multiply rounding and overflow/NaN/warning behavior; the raw public API uses the library's log-space behavior. HCP consumes cached group gates/potentials and shares their fusion. ABIDE retains its training-group gate, selected normalization percentile and alpha scaling. Select each experiment's numerical profile through the shared interface.

HNU and ABIDE share the same native estimator implementation; project-specific edge order, lag/tie policies, dtype, reference selection, group aggregation and directional readouts remain explicit. The upstream `rgic` namespace is isolated from HNU's local `rgic.py`.

Internal validation compared 9,893 original/refactored array entries with explicit shape/dtype/value checks, additional HCP design matrices and one CV fold, and 16 complete main-simulation CSV files byte-for-byte. Independent differential tests covered float16/32/64 arithmetic, warnings/errors, return types, ties, array layout and concurrent calls in two numerical environments. These validation artifacts are retained outside this code-only repository.

Run the dataset-free checks with `python -B run_tests.py`. Use the specified dependency versions for numerical comparisons. Lag direction characterizes observed BOLD temporal ordering; native strength errors compare estimators with different scales.

## Data governance, licensing and citation

Participant measurements, per-person diagnosis records, internal HPC material and saved results are not part of this code-only tree. Input identities/hashes required for reproducibility remain configuration metadata. Obtain and use data under the original providers' conditions; do not post participant-level data in public issues.

The project's original code and documentation are released under the [MIT License](LICENSE). Third-party components retain their applicable licenses, copyright notices and attribution requirements. Source datasets remain subject to their providers' data-use conditions.

| Component | License and attribution |
| --- | --- |
| Project code | MIT; see [LICENSE](LICENSE). |
| PhiID / phyid | Preserve the complete applicable [BSD-3-Clause notice](https://github.com/Imperial-MIND-lab/integrated-info-decomp/blob/6c5f2e9d33c985efbdf875d45cb5a2a6a5cdbf44/LICENSE), also retained at `Simulation/LICENSE`. |
| GCMI reference | The recorded [upstream reference carries GPLv3](https://github.com/robince/gcmi/blob/cd5426bb4c6b7094858f37c3710a4ae9367f8ef1/LICENSE). Applicable upstream copyright and license notices remain in force. |
| MADE and other neural models | The official [MADE-for-ASD source carries Apache-2.0](https://github.com/hasan-rakibul/MADE-for-ASD/blob/main/LICENSE). Model source attributions are recorded in `ABIDE/configs/main/neural_vendor_provenance.json`. |
| Numerical dependencies | External libraries retain their respective licenses. |

Cite the accompanying manuscript and the software version or repository commit used for your analysis. The related method package is [zhangzihan-is-good/IC](https://github.com/zhangzihan-is-good/IC).

For issues, include the command, environment, a minimal synthetic reproducer and expected/actual behavior. Do not include credentials, protected data or participant-linked predictions.

## AI assistance and human responsibility

Development and restructuring used AI-assisted coding for implementation, refactoring, testing and documentation, following the disclosure approach in the [IC method package](https://github.com/zhangzihan-is-good/IC#ai-assistance-and-human-responsibility). Human authors are responsible for reviewing the mathematical definitions, reference-state isolation, input handling, numerical behavior, experimental protocols and scientific interpretation, and for correctness, data governance and release decisions.
