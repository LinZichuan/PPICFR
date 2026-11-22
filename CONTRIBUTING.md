# Extending PPICFR

## Where to make changes

| Component | Implementation |
| --- | --- |
| Post-treatment encoder (PRL) | `next_shareNetwork`; `ShareNetwork` / `TransformerNetwork` in `model/models.py` |
| Pseudo-counterfactual constructor (PCC) | `notshareNetwork`, `all_mu1_network`, `all_mu0_network`; teacher branches in `learn.py` |
| Final counterfactual regressor (CFR) | `shareNetwork`, `mu1_network`, `mu0_network` in `model/models.py` |
| Losses and training | `learn.cal_total_loss()`, `learn.train()` |
| Discrepancy | `model/util.py` |
| Data and splits | `utils.load_data()`, `utils.sample_data()` |
| Evaluation | `eval.py` and `learn.evalWithData()` |

Import model definitions from `model.models`, including `PPICFR` and `PPICFR_on_DRCFR`.

## Workflow

1. Copy an experiment JSON and choose a new result directory.
2. Change the relevant module and add focused tests. Preserve output tuple ordering and document any new configuration keys.
3. Compare against the reference with the same data, environment, and experiment settings.

Keep post-treatment variables as training-time information; a final ITE prediction interface should require only pre-treatment covariates. Changes to losses, gradient routing, seeds, preprocessing, or metrics need separate numerical validation.

Run lightweight tests with:

```bash
python -m unittest discover -s tests -v
```

These cover the entry point, configuration references, and read-only checks, not training accuracy. Issue reports should include the command, code revision, environment, selected configuration, and traceback.
