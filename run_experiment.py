"""Run PPICFR experiment lists with training followed by evaluation."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _load_runtime():
    from learn import effect_estimate
    from eval import evaluation

    try:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    except:
        pass
    return effect_estimate, evaluation


def process_model(model_args):
    effect_estimate, evaluation = _load_runtime()
    print("--------------", flush=True)
    try:
        print(f"Processing model: {model_args.get('name', model_args)}", flush=True)
        effect_estimate(model_args)
    except Exception as e:
        print(f"effect_estimate failed for model {model_args.get('name', model_args)}: {e}", file=sys.stderr, flush=True)
        print("--- Full Traceback ---", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
    try:
        print(f"Evaluating model: {model_args.get('name', model_args)}", flush=True)
        evaluation(model_args)
    except Exception as e:
        print(f"evaluation failed for model {model_args.get('name', model_args)}: {e}", file=sys.stderr, flush=True)
        print("--- Full Traceback ---", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)

def run(experiments):
    _load_runtime()
    with open(experiments, 'r', encoding='utf-8') as f:
        model_args_list = json.load(f)

    print(model_args_list)
    print("models to process:", len(model_args_list), flush=True)

    with ProcessPoolExecutor(max_workers=20) as executor:
        try:
            results = list(executor.map(process_model, model_args_list))
        except KeyboardInterrupt:
            print("Interrupted by user, shutting down...", flush=True)
            executor.shutdown(wait=False)
            sys.exit(1)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments", type=Path,
        help="Experiment JSON path, relative to the repository root; defaults to MODELS_JSON."
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and display the experiment list without training.")
    args = parser.parse_args(argv)
    selected = args.experiments or os.environ.get("MODELS_JSON")
    if not selected:
        parser.error("Select an experiment with --experiments examples/model_ihdp.json")
    path = Path(selected)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        path = path.resolve()
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error("Cannot read experiment list {}: {}".format(path, exc))
    if not isinstance(entries, list) or not entries or any(
        not isinstance(entry, dict) or any(
            not isinstance(entry.get(key), str) or not entry[key]
            for key in ("model_name", "dataset", "config")
        ) or not isinstance(entry.get("params", {}), dict)
        for entry in entries
    ):
        parser.error("Expected a non-empty JSON list with model_name, dataset, config, and optional params.")
    if args.dry_run:
        print("Experiments: {}".format(path))
        print("Working directory: {}".format(PROJECT_ROOT))
        for index, entry in enumerate(entries, start=1):
            print("{}: {} / {} / {}".format(index, entry["model_name"], entry["dataset"], entry["config"]))
        return 0
    os.chdir(PROJECT_ROOT)
    run(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
