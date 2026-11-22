"""Read-only PPICFR preflight; never imports project training code or loads NPZ files."""

import argparse
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_PACKAGES = (
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("scikit-learn", "sklearn"),
    ("geomloss", "geomloss"),
    ("tensorboard", "tensorboard"),
    ("PyYAML", "yaml"),
)
GENERATOR_PACKAGES = (
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("tqdm", "tqdm"),
)
OPTIONAL_PACKAGES = (
    ("causalml", "causalml"),
    ("xgboost", "xgboost"),
    ("POT", "ot"),
    ("catenets", "catenets"),
    ("ot4tee", "ot4tee"),
)
DATASET_CONFIGS = {
    "IHDP": "configs/IHDP/default.yaml",
    "NEWS": "configs/NEWS/default.yaml",
    "synthetic": "configs/Synthetic/default.yaml",
    "real_world_knn": "configs/real_world_knn/default.yaml",
    "real_world_psm": "configs/real_world_psm/default.yaml",
}


def add(issues, level, code, message):
    issues.append({"level": level, "code": code, "message": message})


def package_status(distribution, module):
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        version = None
    try:
        available = importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        available = False
    return available, version


def check_packages(issues, generators=False):
    required = list(CORE_PACKAGES)
    if generators:
        required.extend(GENERATOR_PACKAGES)
    for distribution, module in required:
        available, version = package_status(distribution, module)
        if available:
            add(issues, "info", "package", "{}: {} (discoverable; not imported)".format(
                distribution, version or "version metadata unavailable"))
        else:
            add(issues, "error", "missing_package", "Missing required module {} ({})".format(
                module, distribution))
    for distribution, module in OPTIONAL_PACKAGES:
        available, version = package_status(distribution, module)
        add(issues, "info", "optional_package",
            "{}: {}".format(distribution, version or (
                "available" if available else "not installed (optional)")))
    available, version = package_status("numpy", "numpy")
    if available and version:
        try:
            major = int(version.split(".", 1)[0])
        except ValueError:
            major = None
        if major is not None and major >= 2:
            add(issues, "error" if generators else "warning", "numpy_generator_compatibility",
                "IHDP/NEWS generation requires NumPy < 2; installed: {}.".format(version))


def root_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def check_config_path(root, path, issues):
    if not path.is_file():
        add(issues, "error", "missing_config", "Missing configuration: {}".format(path))
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path.is_file()
    current = root
    for part in relative.parts:
        if part in (".", ".."):
            current = current / part
            continue
        if not current.is_dir():
            break
        try:
            names = {child.name for child in current.iterdir()}
        except OSError as exc:
            add(issues, "error", "path_access", "Cannot inspect {} while checking {}: {}".format(
                current, path, exc))
            return False
        if part not in names:
            alternatives = sorted(name for name in names if name.casefold() == part.casefold())
            if alternatives:
                add(issues, "warning", "path_case",
                    "Case-sensitive portability issue: {} uses {!r}; tracked path uses {!r}. "
                    "No path has been substituted.".format(path, part, alternatives[0]))
                current = current / alternatives[0]
                continue
            break
        current = current / part
    return path.is_file()


def read_yaml(path, yaml_module, issues):
    try:
        content = yaml_module.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml_module.YAMLError) as exc:
        add(issues, "error", "invalid_yaml", "Cannot read {}: {}".format(path, exc))
        return {}
    if not isinstance(content, dict):
        add(issues, "error", "invalid_yaml", "Configuration must be a mapping: {}".format(path))
        return {}
    return content


def check_experiments(root, filename, issues, check_data=False, yaml_module=None):
    path = root_path(root, filename)
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        add(issues, "error", "invalid_experiments", "Cannot read {}: {}".format(path, exc))
        return
    if not isinstance(entries, list) or not entries:
        add(issues, "error", "invalid_experiments", "Experiment JSON must be a non-empty list.")
        return
    add(issues, "info", "experiment_count", "{}: {} configurations, 20 workers.".format(
        path, len(entries)))
    outputs = {}
    for index, entry in enumerate(entries, start=1):
        label = "Entry {}".format(index)
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(key), str) or not entry[key]
            for key in ("model_name", "dataset", "config")
        ):
            add(issues, "error", "invalid_entry",
                "{} needs non-empty string fields model_name, dataset, and config.".format(label))
            continue
        params = entry.get("params", {})
        if not isinstance(params, dict):
            add(issues, "error", "invalid_entry", "{}: params must be a mapping.".format(label))
            continue
        if entry["dataset"] not in DATASET_CONFIGS:
            add(issues, "error", "unknown_dataset", "{}: unsupported dataset {!r}.".format(
                label, entry["dataset"]))
            continue
        files = [root / "configs/default.yaml",
                 root / DATASET_CONFIGS[entry["dataset"]], root_path(root, entry["config"])]
        valid_files = [file for file in files if check_config_path(root, file, issues)]
        if yaml_module is None:
            add(issues, "warning", "yaml_unavailable",
                "{}: merged settings, data paths and device settings were not checked; "
                "PyYAML is unavailable.".format(label))
            continue
        cfg = {}
        for file in valid_files:
            cfg.update(read_yaml(file, yaml_module, issues))
        cfg.update(params)
        cfg["model_name"] = entry["model_name"]
        add(issues, "info", "settings",
            "{}: model={}, teacher={}, device={}, repetitions={}, epochs={}, batch_size={}".format(
                label, cfg["model_name"], cfg.get("teacher_model_name"), cfg.get("device"),
                cfg.get("n_experiments"), cfg.get("epochs"), cfg.get("batch_size")))
        device = str(cfg.get("device", ""))
        if device.startswith("cuda"):
            macos = platform.system() == "Darwin"
            add(issues, "error" if macos else "warning", "cuda_device",
                "{} requests {}: {}".format(label, device,
                    "CUDA is unavailable on macOS." if macos else "verify device availability and memory."))
        selected = (entry["model_name"] + " " + str(cfg.get("teacher_model_name", ""))).lower()
        for token, packages in (
            ("pairnet", (("catenets", "catenets"),)),
            ("perfectmatch", (("catenets", "catenets"),)),
            ("escfr", (("POT", "ot"), ("ot4tee", "ot4tee"))),
            ("xlearner", (("causalml", "causalml"), ("xgboost", "xgboost"))),
            ("rlearner", (("causalml", "causalml"), ("xgboost", "xgboost"))),
            ("drlearner", (("causalml", "causalml"), ("xgboost", "xgboost"))),
            ("crf", (("causalml", "causalml"), ("xgboost", "xgboost"))),
        ):
            if token in selected:
                for distribution, module in packages:
                    if not package_status(distribution, module)[0]:
                        add(issues, "error", "baseline_dependency",
                            "{}: {} requires {}.".format(label, token, distribution))
        if isinstance(cfg.get("verbose"), (int, float)) and cfg["verbose"] <= 0:
            add(issues, "warning", "train_predictions",
                "{}: keep verbose positive; evaluation needs training predictions.".format(label))
        data = cfg.get("data_path")
        if isinstance(data, str) and data:
            data_path = root_path(root, data)
            if not data_path.is_file():
                add(issues, "error" if check_data else "warning", "missing_data",
                    "{}: data file not found: {}".format(label, data_path))
            else:
                add(issues, "info", "data_file", "{}: data exists (not deserialized): {}".format(
                    label, data_path))
        else:
            add(issues, "error", "data_path", "{}: missing/non-string data_path.".format(label))
        output = cfg.get("pred_output_dir")
        if isinstance(output, str) and output:
            try:
                output_path = root_path(root, output).resolve()
            except (OSError, RuntimeError, ValueError) as exc:
                add(issues, "error", "output_path", "{}: cannot resolve output {!r}: {}".format(
                    label, output, exc))
                continue
            if output_path in outputs:
                add(issues, "warning", "shared_output", "{} shares output directory with entry {}: {}".format(
                    label, outputs[output_path], output_path))
            outputs[output_path] = index
            if all((output_path / name).is_file() for name in (
                "eval_result.txt", "eval_result_summary.txt"
            )):
                add(issues, "warning", "cached_results",
                    "{}: results already exist; this run will be skipped: {}".format(label, output_path))
        else:
            add(issues, "error", "output_path", "{}: missing/non-string pred_output_dir.".format(label))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, help="JSON list, relative to the repository root.")
    parser.add_argument("--check-data", action="store_true", help="Treat missing selected data files as errors.")
    parser.add_argument("--generators", action="store_true", help="Also check generator dependencies and NumPy compatibility.")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report.")
    parser.add_argument("--verbose", action="store_true", help="Include package and configuration details.")
    args = parser.parse_args(argv)
    if args.check_data and args.experiments is None:
        parser.error("--check-data requires --experiments")
    issues = []
    check_packages(issues, args.generators)
    if args.experiments is not None:
        try:
            import yaml
        except ImportError:
            yaml = None
        try:
            check_experiments(PROJECT_ROOT, args.experiments, issues, args.check_data, yaml)
        except (OSError, RuntimeError, ValueError) as exc:
            add(issues, "error", "path_access", "Cannot complete path checks for {}: {}".format(
                args.experiments, exc))
    errors = sum(issue["level"] == "error" for issue in issues)
    warnings = sum(issue["level"] == "warning" for issue in issues)
    report = {
        "project": "PPICFR", "project_root": str(PROJECT_ROOT), "python": sys.version.split()[0],
        "platform": platform.platform(), "errors": errors, "warnings": warnings, "issues": issues,
        "scope": "Read-only preflight; no training imports, device probes, NPZ loading, or repairs.",
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("PPICFR preflight | Python {} | {}".format(report["python"], report["platform"]))
        for issue in issues:
            if args.verbose or issue["level"] != "info" or issue["code"] == "experiment_count":
                print("[{}] {}".format(issue["level"].upper(), issue["message"]))
        print("Summary: {} errors, {} warnings.".format(errors, warnings))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
