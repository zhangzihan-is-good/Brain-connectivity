# Reliability-Gated Information Connectivity (RGIC): research code

Research code for **Mapping human brain connectivity through reliable temporal relationships**, by Zihan Zhang, Junhao Gao, Georgios Christopoulos, Xiao Ding, Yuting Chen, Juntong Du, Xia Liang, Bing Qin and Ting Liu. The four experiment folders implement simulated-signal analyses, anatomical-connectivity prediction in HCP, repeated-scan evaluation in HNU and autism classification in ABIDE.

This repository contains scientific code, dependencies, fixed parameters, input specifications, source notices and synthetic checks. Citation metadata is provided in [CITATION.cff](CITATION.cff). Identify the software revision used in an analysis with `git rev-parse HEAD`.

## Repository structure

```text
.
├── README.md
├── LICENSE                    MIT license
├── CITATION.cff               Software citation metadata
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

`FREEENERGY_WORKSPACE` identifies the directory containing an experiment's external data layout. The explicit `--inputs`, `--data-root` and `--data-workspace` options select the locations used by each command below. Write generated outputs to a separate new directory.

## Data inputs and access

There are three kinds of files in this workflow:

- **Source data:** MRI recordings and associated metadata from HCP, HNU1 or ABIDE. Use the official source links in the experiment sections and follow the provider's terms.
- **Study inputs:** source data processed into the exact ROI time series, edge-feature arrays, participant/session order and fixed splits consumed by these programs. These are input data products, distinct from final performance tables or fitted models.
- **Outputs:** new metrics, predictions and models produced when you run an experiment.

To obtain matching study inputs, [open an input-access request](https://github.com/zhangzihan-is-good/Brain-connectivity/issues/new?title=Study%20input%20access%20request) with the maintainers. Specify the experiment, the repository commit from `git rev-parse HEAD`, your operating system/Python version and which files in the table below you need. For ABIDE, also include the resolved local directory where you intend to place the data. Arrange data transfer under the original providers' conditions; do not attach participant data or diagnosis tables to a public issue.

| Experiment | Study inputs to request | Local preparation/check |
| --- | --- | --- |
| HCP | `inputs.npz`, `appendix_inputs.npz`, `sc_weights.npy`, with matching atlas/edge order and fixed folds | [HCP/prepare_inputs.py](HCP/prepare_inputs.py) |
| HNU | 290 preprocessed ROI NPY files and their `input_config.json` participant/session, ROI-order and A/B-group contract | [HNU/prepare_inputs.py](HNU/prepare_inputs.py) |
| ABIDE | The HDF5, phenotype, feature-cache and manifest files listed in [input_contract.json](ABIDE/configs/main/input_contract.json); include the raw ROI inputs for signal-length/noise analyses | [ABIDE/prepare_inputs.py](ABIDE/prepare_inputs.py) |

The preparation tools validate files supplied to them. HCP can additionally copy its three files byte-for-byte; HNU can install the supplied contract for local use. They retain the supplied numerical values, participant mappings and split assignments. The synthetic example above requires none of these study inputs.

## 1. Simulated-signal analyses

**Paper correspondence:** main and supplementary simulated-signal analyses of direction and strength. Inputs are generated from fixed seeds; no external dataset is required.

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

Use the default development/test counts for the article protocol. `--quick` shortens the supplementary run for a smoke check.

## 2. HCP anatomical-connectivity prediction

**Paper correspondence:** anatomical-support/weight prediction, controls, subgroup analyses and supplementary feature/network diagnostics.

Source imaging is available through [HCP Young Adult](https://www.humanconnectome.org/study/hcp-young-adult); accept the relevant data-use terms. Structural-connectome context follows [Liu et al., NeuroImage 2023](https://doi.org/10.1016/j.neuroimage.2023.120276). Request the three study-specific derived files from the maintainers using the input-access procedure above; the HCP and Liu links identify upstream sources, not downloads of these three files.

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

**Paper correspondence:** local/global/high-gain strength, single-scan direction, consensus direction and supplementary support/orientation, roles, degree and participant-specificity readouts.

Obtain source imaging through [HNU1 / CoRR](https://fcon_1000.projects.nitrc.org/indi/CoRR/html/hnu_1.html), [dataset DOI](https://doi.org/10.15387/fcp_indi.corr.hnu1), or the provider's [S3 access instructions](https://fcon_1000.projects.nitrc.org/indi/s3/index.html). Follow the provider's data-use terms. For the paper's input representation, request the 290 preprocessed ROI files and their matching input contract from the maintainers using the input-access procedure above.

The HNU study-input package consists of 290 preprocessed NPY files (29 participants × ten recordings) and a JSON contract. Its `input_contract` describes array shape/dtype/TR; `roi_order` specifies the 109 columns; `halves` records the A/B participant groups; and `recordings` records each file's participant, scan order, session, filename and SHA256. Keep the contract with the files it describes.

Each NPY must be finite float32 with shape **295×109**, matching the contract's identities, ordering and hashes. These inputs have the first five volumes removed and are standardized region-wise. The runner then demeans each signal and applies fourth-order zero-phase Butterworth filtering at 0.01–0.1 Hz with TR=2 seconds.

The first command validates the supplied contract and ROI folder. For the first run in a fresh checkout, also execute the second command: `--install` copies the supplied contract byte-for-byte to the runner's local `HNU/input_config.json`.

```bash
python -B HNU/prepare_inputs.py --contract "HNU_INPUT_CONTRACT" --data-root "HNU_NPY_DIR"
python -B HNU/prepare_inputs.py --contract "HNU_INPUT_CONTRACT" --data-root "HNU_NPY_DIR" --install
```

The tool checks all 290 files and the contract's participant/scan/ROI mapping. An existing different local contract is preserved and reported as an error. The installed participant-linked contract stays local and is excluded by `.gitignore`.

Compute the analyses:

```bash
python -B HNU/run.py --data-root "HNU_NPY_DIR" --output outputs/hnu --workers 3
```

Use a new/empty computation directory. Every run computes primitives from the ROI inputs. Outputs include `summary.csv`, `paired.csv`, `individual_S.csv`, `direction_density.csv` and task NPZ files.

`HNU/models.json` contains the fixed configurations; `common.py` defines scopes, modes, seed and 5,000 whole-participant bootstrap resamples. The input contract assigns participants to group A (14 people) or B (15 people). A-to-B transfer estimates reference reliability from group A scans 1–5 and evaluates group B; B-to-A reverses those roles. The reported combined result gives these two transfers equal weight.

Within either evaluation group, `retest` compares scans 6–10 with one another. `temporal` compares each person's scans 6–10 with that same person's early scans 1–5. For strength, the five early rank profiles are averaged and normalized into an early personal template. Directional-consensus evaluation combines early selected arrows into a fixed template. Both modes use the same selected parameters and frozen reference reliability within a transfer.

The runner evaluates the fixed configurations in `HNU/models.json`. Additional `A_selected_all29` and `B_selected_all29` outputs apply the A-selected or B-selected model to all 29 people; use the held-out-group transfers for the primary analysis.

## 4. ABIDE classification and sensitivity analyses

**Paper correspondence:** autism classification, direction representation, component removal, demographic inputs, low-motion data, signal-length and measurement-noise analyses.

Classifier implementations include SVM, logistic regression, FCN, GCN, LSTM, Transformer and MADE. Fixed refits use the configurations keyed by classifier, atlas, lag window and fold in `ABIDE/configs/reproduction/`; runner options are listed by `--help`.

Obtain ABIDE-I source data from [ABIDE](https://fcon_1000.projects.nitrc.org/indi/abide/) and [preprocessed access](https://fcon_1000.projects.nitrc.org/indi/abide/preprocessed.html), following the provider's terms. This protocol uses CPAC `filt_global` ROI products and a fixed cohort of 1,035 participants (505 ASD, 530 controls).

Request the ABIDE study-input package using the input-access procedure above. [ABIDE/configs/main/input_contract.json](ABIDE/configs/main/input_contract.json) lists its 94 base files and recorded array hashes. `DATA_WORKSPACE` means the local root of this external package, separate from the code checkout. Paths below it are literal keys in the package format:

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

The Chinese directory names are fixed on-disk keys for the input types in this table. Keep those keys when extracting the package. Obtain its source-reference copies and manifests together with the arrays.

The primary files are `dataset/abide_data/04_rgc_intermediate_data/rgc_base.hdf5` and `dataset/abide_data/03_pearson_fc/abide.hdf5`. The base HDF5 uses `patients/{FILE_ID}` with `y` (ASD=0, control=1), `site`, `sex` and `age` attributes. The Pearson file stores the ordered train/validation/test identities at `experiments/{aal,cc200,ez}_whole/{0..9}/{train,valid,test}`. The phenotype CSV is `dataset/abide_data/01_subject_metadata/Phenotypic_V1_0b_preprocessed1.csv`, with `FILE_ID`, `DX_GROUP`, `SEX`, `AGE_AT_SCAN`, `HANDEDNESS_CATEGORY`, `FIQ` and `func_mean_fd` fields.

HDF5/metadata bind participant identities, labels, atlas-specific train/validation/test roles and ROI order. AAL/EZ use 116 regions (6,670 pairs); CC200 uses 200 (19,900 pairs). The checker accepts a package when its NPZ/validation JSON, aligned NPY arrays, metadata/source recipes, runtime versions and identity hashes match the fixed contracts.

The native-cache reader hashes the two source-reference files named in the table. Identical source bytes are included as `estimators_shared.py` and `ABIDE/code/main/source_tree/freeenergy/第四部分/code/prepare_raw.py`. A recipe manifest records source hashes, runtime versions, array identities and absolute source-path keys. The checker compares the resolved workspace root with those recorded keys. If they differ, use the declared placement or arrange a matching package with the maintainers; include your intended resolved root in the request. Editing path strings or merely exposing a directory alias does not establish a matching recipe.

Signal-length/noise analyses additionally require `第四部分/ABIDE后续分类器队列_20261027/inputs/original_cohort_and_folds.json` and the AAL recordings at `第四部分/inputs/raw_timeseries/aal/{FILE_ID}_rois_aal.1D`, including the matching ROI/header/TR metadata specified by the input contract.

Preflight the package before fitting:

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

Refit one fixed main unit using a new/empty output directory outside the code and protected data locations:

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

Cite Zihan Zhang et al., *Mapping human brain connectivity through reliable temporal relationships*, together with **Brain-connectivity: RGIC research code**, its repository URL and the Git commit used for your analysis. [CITATION.cff](CITATION.cff) supplies the software authors and citation metadata. The related method package is [zhangzihan-is-good/IC](https://github.com/zhangzihan-is-good/IC).

For issues, include the command, environment, a minimal synthetic reproducer and expected/actual behavior. Do not include credentials, protected data or participant-linked predictions.

## AI assistance and human responsibility

Development and restructuring used AI-assisted coding for implementation, refactoring, testing and documentation, following the disclosure approach in the [IC method package](https://github.com/zhangzihan-is-good/IC#ai-assistance-and-human-responsibility). Human authors are responsible for reviewing the mathematical definitions, reference-state isolation, input handling, numerical behavior, experimental protocols and scientific interpretation, and for correctness, data governance and release decisions.
