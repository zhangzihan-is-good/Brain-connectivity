# Reliability-Gated Information Connectivity (RGIC): research code

Code for the simulation, HCP, HNU and ABIDE analyses accompanying the RGIC manuscript. This repository contains scientific code, dependencies, fixed parameters, input contracts, source notices and synthetic checks.

## Repository structure

```text
.
├── README.md
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
| Synthetic example and dataset-free checks | Python 3.12, `python -m pip install -r requirements-demo.txt` (NumPy 2.3.5). CPU only. |
| Simulation | Python 3.12 target, `python -m pip install -r Simulation/requirements.txt`. CPU; worker count is configurable. |
| HCP | Python 3.12 target, `python -m pip install -r HCP/requirements.txt`. CPU. |
| HNU | Python 3.9 and `python -m pip install -r HNU/requirements.txt`. CPU; optional exact-summation acceleration has a fallback. |
| ABIDE refitting | The archived Windows Python 3.9.7 classifier environment: NumPy 1.26.4, SciPy 1.10.1, scikit-learn 1.4.2, h5py 3.6.0 and the other dependencies in `ABIDE/requirements.txt`. Neural workflows use the specified PyTorch 2.5.1+cu121 environment. |
| ABIDE plotting | `ABIDE/code/figures/requirements.txt`; a font with the necessary CJK glyphs is required for the existing plot labels. |

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

SVM/LR main refits use CPU. Neural model device settings are retained; the supplementary FCN workflow explicitly requires a compatible NVIDIA GPU/driver. No VRAM guarantee is made.

Actual numerical comparisons used existing Windows environments. Historical simulation/HCP regression used Python 3.9.7, NumPy 1.21.5, SciPy 1.10.1, pandas 1.3.0 and scikit-learn 1.4.2. HNU used its historical environment. ABIDE comparisons included its NumPy 1.26.4 environment and a separate NumPy 2.3.5 environment. Old/new code was compared within the same environment.

These comparisons are distinct from a fresh-install test of every target requirement set. Fresh-download install durations and complete Linux/macOS runs have not been benchmarked. Keep the specified versions when reproducing a recorded numerical comparison.

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

Measured computation time was about 0.025 seconds, excluding interpreter startup, in the tested Python 3.12.14 / NumPy 2.3.5 environment. This is a library demonstration, not the paper benchmark or a clinical example.

`run_tests.py` checks the selected IC source hashes, the synthetic/shared-interface tests and upstream scientific/file-roundtrip tests. It uses no original sibling checkout, participant data, saved regression fixture or GPU. `run_public_tests.py` runs only the synthetic/shared-interface subset.

## Path and output conventions

All commands below are executed from the repository root. Replace input placeholders with actual paths. Direct script commands retain the current working directory, so the relative `outputs/...` paths below are consistent between computation and plotting.

The optional common launcher is:

```bash
python -B run.py info
python -B run.py hcp -- --help
python -B run.py --python /path/to/interpreter hnu -- --help
```

It executes a selected runner in that experiment's directory. Supply **absolute input/output paths** when using this launcher. Arguments following `--` are forwarded unchanged.

`FREEENERGY_WORKSPACE` can identify the external workspace containing `dataset/`; explicit `--inputs`, `--data-root` and `--data-workspace` parameters are preferable for independent installations. Do not write new results over original data, source, caches or historical runs.

## 1. Simulated-signal analyses

**Paper correspondence:** main direction/strength experiments and the two simulation supplementary figures. Inputs are generated from fixed seeds; no external dataset is required.

The fixed protocol has two acquisition profiles, three degradation settings and six relationship types (36 conditions). Each world has 24 pairs, 16 reference recordings and three target recordings. There are four development and ten test repeats per condition. The main direction cohort has 360 test worlds, and native strength uses 300 separately generated worlds.

RGIC uses reference-only F×L×S, a physical lag window up to 2.2 seconds, lambda=2 and gamma=0.45 in inverse seconds. W2 uses ±1/±2 sampling offsets with lambda=2.25 and gamma=0.55. Constants, conditions and seeds are defined in `Simulation/core.py`; fixed experiments and statistics are in `Simulation/experiment.py`.

Run the main analysis and figure:

```bash
python -B Simulation/run.py --workers 4 --output outputs/simulation/main --dpi 600
```

This computes development thresholds and grids, test direction/AP, native strength and component ablations, including the shared-measurement-noise setting. Tables are written under `outputs/simulation/main/results/`; the statistical figure is `Figure1_updated.pdf/png/svg` under `figures/`.

Run the supplementary analyses:

```bash
python -B Simulation/appendix_experiment.py --workers 4 --output outputs/simulation/appendix
python -B Simulation/appendix_figures.py --input outputs/simulation/appendix/results --output outputs/simulation/appendix/figures --dpi 600
```

These cover signal spectra/tSNR/autocorrelation, lag peaks, W2, profile-specific AP, lag error, noise/knee sensitivity, nearest-neighbor TE, development-selected classical settings, PSI standardization and reference-supported direction scores. Outputs include `Appendix1Figure1/2` PDF/PNG/SVG and source tables. Four-pair expensive-comparator and full-24-pair reference summaries are separate outputs.

A `--quick` supplementary run or changed development/test counts is a smoke run, not the article protocol. The main calculation took approximately 4.6–4.8 minutes in local four-thread audits using the historical environment; other hardware/environments and the full supplementary runtime were not benchmarked. Manuscript schematic artwork is separate from the statistical renderer.

## 2. HCP anatomical-connectivity prediction

**Paper correspondence:** anatomical-support/weight prediction, controls, subgroup analyses and supplementary feature/network diagnostics.

Source imaging is available through [HCP Young Adult](https://www.humanconnectome.org/study/hcp-young-adult); accept the relevant data-use terms. Structural-connectome context follows [Liu et al., NeuroImage 2023](https://doi.org/10.1016/j.neuroimage.2023.120276). Public imaging downloads alone do not generate this project's required derived inputs.

Provide one directory containing:

| File | Required content |
| --- | --- |
| `inputs.npz` | Group RGIC descriptors, seven native comparator vectors, upper-triangle edge identities, structural support, network labels, fixed CV assignments and `metadata_json`. |
| `appendix_inputs.npz` | The retained group features and edge/CV/network information used by supplementary diagnostics. |
| `sc_weights.npy` | Symmetric 400×400 structural weight matrix with matching region order. |

The contract uses 400 regions, 79,800 unordered upper-triangle edges, 5,059 positive structural-support labels and 20×79,800 fixed fivefold assignments. Group summaries represent 33 participants × four recordings. W1 and lambda=gamma=1 in sampling units are retained. Both NPZ files require `metadata_json.fixed_configuration`; the main input also requires `metadata_json.configuration_origin`. Supplementary network labels must be encoded exactly as integers 0–6. Cache arrays, metadata and region/edge order must match the checks in `HCP/evaluate.py` and `HCP/supplement_diagnostics.py`.

Compute and render using a new or empty output directory:

```bash
python -B HCP/run.py --inputs "HCP_INPUT_DIR" --output outputs/hcp --workers 3 --panels
```

Separate computation and plotting when needed:

```bash
python -B HCP/run.py --inputs "HCP_INPUT_DIR" --output outputs/hcp --compute-only
python -B HCP/run.py --inputs "HCP_INPUT_DIR" --output outputs/hcp --plot-only --panels
```

`--section main` and `--section appendix` restrict the run. Main outputs include `fold_results.csv`, `node_repeat_results.csv`, `weight_repeat_results.csv`, `control_repeat_results.csv`, `stratified_results.csv` and `summary.json`. The main statistical renderer writes `statistical_results.png/svg`, plus optional panel PNGs. Supplementary outputs include feature contrasts, coefficients, network-pair metrics and `RGIC_HCP_supplementary.pdf/png/svg`.

This workflow starts from group-level caches; it does not provide the raw-BOLD preprocessing/feature-generation and group-aggregation pipeline. It does not re-estimate cached gates or potentials. All classification models retain distance and absolute group Pearson correlation; the full RGIC model has 17 predictors. Edge CV is not participant-held-out CV, and repeated-partition SD is not a participant-level interval. Final workflow artwork/panel assembly is separate.

## 3. HNU test–retest analyses

**Paper correspondence:** local/global/high-gain strength, single-scan direction, consensus direction and supplementary support/orientation, roles, degree and participant-specificity readouts.

Obtain source data through [HNU1 / CoRR](https://fcon_1000.projects.nitrc.org/indi/CoRR/html/hnu_1.html), [dataset DOI](https://doi.org/10.15387/fcp_indi.corr.hnu1). Follow the provider's registration and data terms.

The computation requires 290 preprocessed NPY files: the exact 29 participants × ten recordings in `HNU/input_config.json`. Each file must be finite float32 with shape **295×109**, matching filenames, participant/session identities, A/B assignments, region order and SHA256 values. The supplied series have the first five volumes removed and were standardized region-wise. The runner applies the original demeaned 0.01–0.1 Hz fourth-order zero-phase Butterworth filtering with TR=2 seconds.

Compute, then plot the new results:

```bash
python -B HNU/run.py --data-root "HNU_NPY_DIR" --output outputs/hnu --workers 3
python -B HNU/plot.py --summary outputs/hnu/summary.csv --paired outputs/hnu/paired.csv --output outputs/hnu/figures
python -B HNU/plot_supplement.py --summary outputs/hnu/summary.csv --density outputs/hnu/direction_density.csv --output outputs/hnu/supplement
```

Use a new/empty computation directory. Every run computes primitives from the ROI inputs; historical NPZ output is not an input. Outputs include `summary.csv`, `paired.csv`, `individual_S.csv`, `direction_density.csv`, task NPZ files and the requested figures.

`HNU/models.json` contains the fixed configurations; `common.py` defines scopes, modes, seed and 5,000 whole-participant bootstrap resamples. Reference reliability uses training-group scans 1–5 and is frozen for the opposite group. The two primary transfers are evaluated separately and averaged equally. Later-only and early-template comparisons share the selected model within a transfer.

The code reproduces fixed selected models, not the original six-fold development search/candidate-pool selection. Exploratory all-participant selector views do not replace held-out-group transfers. The available main plot layout and the final manuscript assembly are distinct.

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

HDF5/metadata bind participant identities, labels, atlas-specific train/validation/test roles and ROI order. AAL/EZ use 116 regions (6,670 pairs); CC200 uses 200 (19,900 pairs). The approved derived inputs include the original NPZ/validated JSON under `第四部分/inputs`, and the `report_base` and `native_1035` atlas-specific NPY/manifest bundles under `第四部分/按论文报告复现_20261027/inputs`. Derived arrays, metadata/source recipes, runtime and identity hashes must agree. Public downloads alone do not reconstruct every frozen derived cache.

The native-cache reader also hashes `DATA_WORKSPACE/第三部分最终版/code/estimators.py` and `DATA_WORKSPACE/第四部分/code/prepare_raw.py`; it does not execute those external files on the cached path. Identical source bytes are included as `estimators_shared.py` and `ABIDE/code/main/source_tree/freeenergy/第四部分/code/prepare_raw.py`. Preserve the external input bundle's associated source copies and recipe bindings. Historical recipes bind absolute path strings as well as source/runtime hashes, so relocating a bundle is not automatically portable; changing or bypassing these checks is outside the validated protocol.

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

The default fits the methods for that unit; `--method RGIC` selects RGIC alone. Generated outputs include the protocol, per-method predictions/scores/metrics and model files. The fixed refit does not start another broad parameter search.

Run selected supplementary experiments:

```bash
python -B ABIDE/run.py --mode supplements --execute --data-workspace "DATA_WORKSPACE" --output "NEW_ABIDE_SUPPLEMENTS" --rebuild-raw
```

`--family`, `--classifier` and `--fold` restrict model tasks. The complete selection has 5,720 fold records in 572 groups. The task configuration retains fixed conditions and identity-hash checks; the runner derives participant axes from the approved external metadata rather than shipping participant labels. The low-motion subset requires finite `0 <= func_mean_fd <= 0.2` and filters the original roles without a new split. Existing raw caches bound to different source recipes must not be silently reused. `--rebuild-raw` builds declarations/inputs in the new output.

After all selected records complete, summarize the same output directory to produce the plotting input. Summaries use 500 conditional paired bootstrap resamples with seed 20261030:

```bash
python -B ABIDE/run.py --mode supplements --execute --action summarize --data-workspace "DATA_WORKSPACE" --output "NEW_ABIDE_SUPPLEMENTS"
```

Render results you have generated:

```bash
python -B ABIDE/run.py --mode figures --input "NEW_ABIDE_SUPPLEMENTS/reports/selected_AUROC_results.json" --verify-data
python -B ABIDE/run.py --mode figures --input "NEW_ABIDE_SUPPLEMENTS/reports/selected_AUROC_results.json" --output "NEW_ABIDE_FIGURES"
```

The renderer produces the selected S1/S3/S4/S5 PDF/PNG/SVG series. It requires explicit result input; no historical result table or figure is distributed. The final manuscript figure's artwork and panel assembly remain a separate step.

**Interpretation:** supervised F screening uses train, validation and test labels. This reproduces the original protocol; the resulting AUROC is not an independent leakage-free generalization estimate. Only RGIC received the broad validation search, with comparator methods using matched selected settings. Demographics enter the classifier, not the RGIC formula. Fixed feature columns still involve classifier refitting; truncation/noise experiments do not mean applying a clean-trained model unchanged.

## Shared RGIC implementation and validation

IC version `0.1.0a3` is pinned at [58625fc4a3afd69d451f27b5c279a3c2a3cae90f](https://github.com/zhangzihan-is-good/IC/commit/58625fc4a3afd69d451f27b5c279a3c2a3cae90f). `_vendor/IC_PROVENANCE.json` records the **selected** unmodified runtime source, required CLI resource, notice and synthetic upstream checks. It is not a complete copy of the upstream repository.

The required CLI manual resource retains its upstream text and hash. Installation and example paths in that resource refer to the complete upstream IC checkout; use this root README's commands for the selected source bundled here.

The adapter calls the pinned private `_gate` and `_potential` primitives while preserving experimental conventions. Direct compensation profiles retain the historical exp/multiply rounding and overflow/NaN/warning behavior; the raw public API uses the library's log-space behavior. HCP only consumes cached group gates/potentials and shares their fusion. ABIDE retains its training-group gate, selected normalization percentile and alpha scaling. These experiments are not interchangeable with a default public `RGIC.fit/transform` call.

HNU and ABIDE share the same native estimator implementation; project-specific edge order, lag/tie policies, dtype, reference selection, group aggregation and directional readouts remain explicit. The upstream `rgic` namespace is isolated from HNU's local `rgic.py`.

Internal validation compared 9,893 original/refactored array entries with explicit shape/dtype/value checks, additional HCP design matrices and one CV fold, and 16 complete main-simulation CSV files byte-for-byte. Independent differential tests covered float16/32/64 arithmetic, warnings/errors, return types, ties, array layout and concurrent calls in two numerical environments. These validation artifacts are retained outside this code-only repository.

The dataset-free checks are reproducible here. They do not prove all-input/all-platform equivalence, nor replace full HCP repeated CV, all HNU final outputs or all classifier training. Dependency-version changes require same-environment comparisons. Lag direction describes observed BOLD timing rather than established neural causality; native strength errors compare estimators with different scales.

## Data governance, licensing and citation

Participant measurements, per-person diagnosis records, internal HPC material and saved results are not part of this code-only tree. Input identities/hashes required for reproducibility remain configuration metadata. Obtain and use data under the original providers' conditions; do not post participant-level data in public issues.

A repository-wide open-source license has not yet been designated, and the pinned IC version lacks an explicit license grant. `Simulation/LICENSE` preserves a component-specific phyid BSD-3-Clause notice; it does not license all project code. The applicable provenance/permissions for third-party adaptations must be resolved before public distribution. This README does not declare an open-source clearance.

| Component | Source and release responsibility |
| --- | --- |
| Authors' experiment code and pinned IC | An authorized rights holder must designate the project license and the applicable permission for the pinned IC version. Public visibility, citations and source hashes do not grant redistribution rights. |
| PhiID / phyid | Preserve the complete applicable [BSD-3-Clause notice](https://github.com/Imperial-MIND-lab/integrated-info-decomp/blob/6c5f2e9d33c985efbdf875d45cb5a2a6a5cdbf44/LICENSE), also retained at `Simulation/LICENSE`. |
| GCMI | The recorded [upstream version is GPLv3](https://github.com/robince/gcmi/blob/cd5426bb4c6b7094858f37c3710a4ae9367f8ef1/LICENSE). Establish whether the local compact implementation derives from source code or independently from mathematical descriptions, then preserve the applicable terms. |
| MADE and other neural models | The official [MADE-for-ASD source currently carries Apache-2.0](https://github.com/hasan-rakibul/MADE-for-ASD/blob/main/LICENSE). Establish the adopted version and Torch-port provenance and preserve applicable licenses/notices. For every other model, confirm authorship or adaptation rights; extraction hashes alone do not establish permission. |
| Numerical dependencies | Requirements install external libraries. Review their applicable terms separately if distributing binaries, containers or bundled runtimes. |

Cite the accompanying manuscript and its associated released software version/commit. Final publication and software DOI/tag metadata should be supplied by the authors rather than invented. The related method package is [zhangzihan-is-good/IC](https://github.com/zhangzihan-is-good/IC).

For issues, include the command, environment, a minimal synthetic reproducer and expected/actual behavior. Do not include credentials, protected data or participant-linked predictions.

## AI assistance and human responsibility

Development and restructuring used AI-assisted coding for implementation, refactoring, testing and documentation, following the disclosure approach in the [IC method package](https://github.com/zhangzihan-is-good/IC#ai-assistance-and-human-responsibility). Human authors review the mathematical definitions, reference-state isolation, input handling, numerical behavior, experimental protocols and scientific interpretation, and remain responsible for correctness, data governance and release decisions, and remains responsible for the code, its scientific interpretation, and release decisions.