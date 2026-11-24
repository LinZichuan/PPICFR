# Using PPICFR

## Running experiments

```bash
python run_experiment.py --experiments examples/model_ihdp.json
python run_experiment.py --experiments examples/model_ihdp.json --dry-run
```

Relative paths resolve from the repository root. `--dry-run` validates and displays the JSON list without importing training dependencies. Environment-variable selection is also supported:

```bash
MODELS_JSON=examples/model_news.json python run_experiment.py
```

`--experiments` takes precedence over `MODELS_JSON`; an experiment list must be selected explicitly. The data-generation blocks in the [README](../README.md) are run from the repository root with `cd`; `run_experiment.py` also changes to the repository root, so relative paths such as `./examples/...` resolve consistently.

| Example | Configurations | Repetitions | Epoch limit | Device |
| --- | ---: | ---: | ---: | --- |
| `model_ihdp.json` | 1 | 50 | 100 | CPU |
| `model_news.json` | 1 | 50 | 100 | CPU |
| `model_synthetic.json` | 20 | 50 | 1000 | `cuda:0`–`cuda:7` |

The runner processes configurations with a 20-worker pool. A dataset is split 60/20/20 for training/validation/test by default.

## Configuration

Each JSON entry selects a model, dataset, configuration file, and optional overrides:

```json
[
  {
    "model_name": "ppicfr",
    "dataset": "IHDP",
    "config": "./configs/IHDP/ppicfr.yaml",
    "params": {
      "device": "cpu",
      "pred_output_dir": "results/my_experiment"
    }
  }
]
```

Copy an example to a new JSON file to create your own experiment. Configuration precedence, from lowest to highest, is:

`configs/default.yaml` → dataset `default.yaml` → selected model YAML → `params` → JSON `model_name`.

Dataset values are `IHDP`, `NEWS`, and `synthetic`; model configuration files are in `configs/<dataset>/`.

| Setting | Purpose |
| --- | --- |
| `data_path`, `pred_output_dir` | Data input and result directory |
| `n_experiments`, `epochs`, `batch_size` | Repetition count and training size |
| `lr`, `optim`, `l2`, `do_rate` | Optimizer, regularization, and dropout |
| `teacher_model_name` | Pseudo-counterfactual constructor backbone |
| `use_transformer` | Post-treatment encoder selection |
| `hs_kl_w` | KL loss weight |
| `hs_imb_dist`, `hs_imb_dist_w` | `mmd`/`wass` discrepancy and weight |
| `valid_metrics` | Model selection criterion; `pehe` requires true effects |
| `fix_testset` | Whether test membership is fixed across repetitions |

Keep `verbose` positive when using `eval.py`, which reads the training-set prediction files.

Synthetic generator arguments are sequence length and noise scale:

```bash
(cd data/synthetic && python generate_data.py 60 1.0)
```

Data filenames encode their generation settings. After generation, check which data files the Synthetic matrix expects with:

```bash
python tools/check_environment.py --experiments examples/model_synthetic.json --check-data
```

IHDP/NEWS generators use the first bundled benchmark replicate and generate the temporal variables and outcomes.

## Data format

The NPZ loader expects:

| Field | Meaning |
| --- | --- |
| `x` | Pre-treatment covariates |
| `t` | Binary treatment |
| `yf`, `ycf` | Factual and counterfactual outcomes |
| `tau` | Ground-truth treatment effect |
| `next_x` | Flattened post-treatment observations |
| `num_timesteps`, `m_x`, `m_v`, `m_m`, `m_a` | Temporal layout metadata |
| `tau_original`, `y_scaler` | Optional original-scale effects and outcome scaler |

Bundled arrays use `[N,D,1]` for features and `[N,1]` for treatments/outcomes. Splitting squeezes singleton dimensions. Synthetic post-treatment features use consecutive V/M/A blocks, converted to `[N,T,m_v+m_m+m_a]` for the encoder.

The fields `ycf` and `tau` are used for validation and evaluation; the bundled generators provide them. PPICFR assumes binary treatment, overlap, SUTVA, and unconfoundedness. Models are defined in `model/layers.py` (shared components), `model/ppicfr.py` (PPICFR), and `model/baselines.py` (baselines); the training loop is in `learn.py`. NPZ files may contain pickled objects, so load only trusted files.

## Results

The selected output directory contains:

- `ppicfr_{train,valid,test}_result.test.npz`: predictions and losses;
- `eval_result.txt`, `eval_result_summary.txt`: metric summaries.

Prediction arrays use `[sample position, repetition, output slot]`. NPZ files contain predictions, not model checkpoints. If some results are missing, the console output and `filelogs/running_time.log` show the failing configuration and its traceback.

If both evaluation text files exist, training loops and metric recomputation are skipped; the configured data file is still loaded. Use a fresh `pred_output_dir` for a new configuration. In `eval_result_summary.txt`, `±` denotes the standard error across repetitions.

## Environment checks

```bash
python tools/check_environment.py --generators
python tools/check_environment.py --experiments examples/model_news.json --check-data
python tools/check_environment.py --experiments examples/model_news.json --json
```

Checks report missing packages, configuration/data paths, device requests, and existing result caches without running training or loading NPZ contents. Exit codes: `0` no errors, `1` blockers, `2` invalid arguments. Add `--verbose` for package details.

## Baseline dependencies

| Branch | Dependencies |
| --- | --- |
| Meta-learners / causal forest | `causalml`, `xgboost` |
| PairNet / matching | `catenets`, JAX; follow [PairNet installation](https://github.com/nlokeshiisc/pairnet_release) |
| ESCFR | `ot4tee`, `POT` (imported as `ot`) |

Missing baseline packages do not affect the default PPICFR experiment.
