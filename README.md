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

## Run IHDP

Execute from the repository root:

```bash
(cd data/IHDP && python generate_data.py)
python run_experiment.py --experiments examples/model_ihdp.json
```

This runs 50 repetitions, with up to 100 epochs each, on CPU. Results are saved to `results/IHDP/`.

To inspect an experiment without training:

```bash
python run_experiment.py --experiments examples/model_ihdp.json --dry-run
python tools/check_environment.py --experiments examples/model_ihdp.json --check-data
```

## Other datasets

```bash
# NEWS
(cd data/NEWS && python generate_data.py)
python run_experiment.py --experiments examples/model_news.json

# Synthetic: generate the configured sequence-length/noise settings
(cd data/synthetic && bash generate_data.sh)
python run_experiment.py --experiments examples/model_synthetic.json
```

The Synthetic list runs 20 configurations on `cuda:0`–`cuda:7`. To run a single configuration on CPU, copy an entry and set `device` in `params` (see [configuration](docs/USAGE.md#configuration)).

## Project structure

| Path | Purpose |
| --- | --- |
| `data/` | Benchmark inputs and temporal data generators |
| `configs/` | Global, dataset, and PPICFR configuration files |
| `examples/` | JSON experiment lists |
| `model/models.py` | PPICFR, encoders, outcome heads, and teacher networks |
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
