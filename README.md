# PPICFR: Post-treatment-informed Pseudo-outcome Imputation for Individual Treatment Effect Estimation

Code for **WSDM 2027**. [Paper](https://doi.org/10.1145/3862322.3863512) · [arXiv](https://arxiv.org/abs/2512.18737) · [Usage](docs/USAGE.md) · [Contributing](CONTRIBUTING.md)

PPICFR uses post-treatment variables to construct pseudo-counterfactual outcomes during training. The final treatment-effect estimator uses only pre-treatment covariates. This repository includes IHDP, NEWS, and synthetic temporal experiments.

## Install

Python 3.10 or 3.11 is recommended. IHDP/NEWS data generation requires NumPy < 2.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt 'numpy<2'
python tools/check_environment.py --generators
```

PairNet and ESCFR need [additional dependencies](docs/USAGE.md#baseline-dependencies).

## Generate Training Data

**IHDP**
```
cd data/IHDP
python generate_data.py
cd ..
```

**NEWS**
```
cd data/NEWS
python generate_data.py
cd ..
```

**Synthetic Data**
```
cd data/synthetic
bash generate_data.sh
cd ..
```

## Run Experiments

Run experiments for NEWS dataset
```
MODELS_JSON='./examples/model_news.json' python run_experiment.py
```

Run experiments for IHDP dataset
```
MODELS_JSON='./examples/model_ihdp.json' python run_experiment.py
```

Run experiments for Synthetic dataset
```
MODELS_JSON='./examples/model_synthetic.json' python run_experiment.py
```

`MODELS_JSON` is the configure information of models. You can add model in `./examples/`.

## Documentation

- **[Usage](docs/USAGE.md)** is the reference for running and troubleshooting experiments. It covers how experiment lists are selected (`MODELS_JSON` and `--experiments`), the JSON/YAML configuration keys and their precedence, the NPZ data format, result files, read-only environment checks, extra baseline dependencies, and the size and device requirements of each example list (the Synthetic list runs 20 configurations on `cuda:0`–`cuda:7`). Use it when adding a JSON file under `examples/`, tuning `configs/`, reading `results/`, or checking inputs before a long run, e.g. `python tools/check_environment.py --experiments ./examples/model_ihdp.json --check-data`.
- **[Contributing](CONTRIBUTING.md)** is the guide for extending the code. It maps each component (post-treatment encoder, pseudo-counterfactual constructor, final counterfactual regressor, baselines, losses, discrepancy, data loading, evaluation) to its implementation, and gives the change workflow, the lightweight test command, and what an issue report should include. Follow it before changing a model, loss, seed, preprocessing, or metric; run `python -m unittest discover -s tests -v` before proposing a change.

## Project structure

| Path | Purpose |
| --- | --- |
| `data/` | Benchmark inputs and temporal data generators |
| `configs/` | Global, dataset, and PPICFR configuration files |
| `examples/` | JSON experiment lists |
| `model/layers.py` | Shared network components |
| `model/ppicfr.py` | PPICFR models |
| `model/baselines.py` | Baseline architectures and estimators |
| `learn.py` | Losses, model construction, and training |
| `eval.py` | Metrics and result summaries |
| `utils.py` | Configuration loading, data loading, and splitting |
| `run_experiment.py` | Experiment entry point |
| `tools/check_environment.py` | Read-only environment and input checks |

## Citation

```bibtex
@inproceedings{lin2027ppicfr,
  title = {PPICFR: Post-treatment-informed Pseudo-outcome Imputation for Individual Treatment Effect Estimation},
  author = {Lin, Zichuan and Huang, Xiaokai and Han, Yuxuan and Ren, Yixin and Liu, Jiate and Chen, Jia and Wu, Xiapeng and Ye, Deheng},
  booktitle = {Proceedings of the Twentieth ACM International Conference on Web Search and Data Mining},
  year = {2027},
  doi = {10.1145/3862322.3863512}
}
```

## License

[Apache-2.0](LICENSE). Datasets and external baseline implementations retain their own licenses.
