"""Tests for additive PPICFR tooling, without numerical dependencies or training."""

import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load_module("ppicfr_runner_test", ROOT / "run_experiment.py")
checker = load_module("ppicfr_checker_test", ROOT / "tools/check_environment.py")


def yaml_stub(configs):
    return types.SimpleNamespace(safe_load=lambda text: configs[text], YAMLError=ValueError)


class RunnerTests(unittest.TestCase):
    def test_dry_run_does_not_import_runtime_or_change_state(self):
        before, cwd = dict(os.environ), Path.cwd()
        with mock.patch.object(launcher, "run") as run, \
                mock.patch.object(launcher, "_load_runtime") as runtime, \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(launcher.main(["--experiments", "examples/model_ihdp.json", "--dry-run"]), 0)
        run.assert_not_called()
        runtime.assert_not_called()
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(Path.cwd(), cwd)
        self.assertIn(str(ROOT / "examples/model_ihdp.json"), out.getvalue())

    def test_cli_precedence_and_root_before_execution(self):
        calls = []
        with mock.patch.dict(os.environ, {"MODELS_JSON": "examples/model_news.json"}), \
                mock.patch.object(launcher.os, "chdir", side_effect=lambda path: calls.append(("cwd", path))), \
                mock.patch.object(launcher, "run", side_effect=lambda path: calls.append(("run", path))):
            self.assertEqual(launcher.main(["--experiments", "examples/model_ihdp.json"]), 0)
            self.assertEqual(os.environ["MODELS_JSON"], "examples/model_news.json")
        self.assertEqual(calls, [("cwd", ROOT), ("run", ROOT / "examples/model_ihdp.json")])

    def test_environment_selects_list(self):
        with mock.patch.dict(os.environ, {"MODELS_JSON": "examples/model_news.json"}), \
                mock.patch.object(launcher.os, "chdir"), mock.patch.object(launcher, "run") as run:
            self.assertEqual(launcher.main([]), 0)
        run.assert_called_once_with(ROOT / "examples/model_news.json")

    def test_relative_list_from_another_directory(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                with mock.patch.object(launcher, "run") as run, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(launcher.main(["--experiments", "examples/model_ihdp.json", "--dry-run"]), 0)
                run.assert_not_called()
                self.assertEqual(Path.cwd(), Path(directory))
            finally:
                os.chdir(previous)

    def test_missing_selection_or_file_fails_before_runtime(self):
        for arguments in ([], ["--experiments", "examples/absent.json"]):
            with self.subTest(arguments=arguments), mock.patch.dict(os.environ, {}, clear=True), \
                    mock.patch.object(launcher, "_load_runtime") as runtime, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                launcher.main(arguments)
            self.assertEqual(raised.exception.code, 2)
            runtime.assert_not_called()

    def test_invalid_json_fails_before_execution(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "invalid.json"
            for content in ("{invalid", "[]", "{}", '[{"model_name":"ppicfr"}]',
                            '[{"model_name":"ppicfr","dataset":"IHDP","config":"x","params":[]}]'):
                path.write_text(content, encoding="utf-8")
                with mock.patch.object(launcher, "run") as run, contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit) as raised:
                    launcher.main(["--experiments", str(path), "--dry-run"])
                self.assertEqual(raised.exception.code, 2)
                run.assert_not_called()

    def test_pool_size_and_configuration_order_are_preserved(self):
        path = ROOT / "examples/model_synthetic.json"
        entries = json.loads(path.read_text())
        with mock.patch.object(launcher, "_load_runtime"), \
                mock.patch.object(launcher, "ProcessPoolExecutor") as pool, \
                contextlib.redirect_stdout(io.StringIO()):
            executor = pool.return_value.__enter__.return_value
            executor.map.return_value = iter([None] * len(entries))
            launcher.run(path)
        pool.assert_called_once_with(max_workers=20)
        executor.map.assert_called_once_with(launcher.process_model, entries)

    def test_train_then_evaluate_without_parameter_changes(self):
        entry = {"model_name": "ppicfr", "dataset": "IHDP", "config": "configs/IHDP/ppicfr.yaml"}
        before = dict(entry)
        order = []
        train = mock.Mock(side_effect=lambda args: order.append(("train", args)))
        evaluate = mock.Mock(side_effect=lambda args: order.append(("evaluate", args)))
        with mock.patch.object(launcher, "_load_runtime", return_value=(train, evaluate)), \
                contextlib.redirect_stdout(io.StringIO()):
            launcher.process_model(entry)
        self.assertEqual(order, [("train", entry), ("evaluate", entry)])
        self.assertIs(train.call_args.args[0], entry)
        self.assertEqual(entry, before)

    def test_training_failure_still_attempts_evaluation(self):
        train, evaluate = mock.Mock(side_effect=ValueError("failed")), mock.Mock()
        with mock.patch.object(launcher, "_load_runtime", return_value=(train, evaluate)), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
            launcher.process_model({"model_name": "ppicfr"})
        evaluate.assert_called_once()
        self.assertIn("effect_estimate failed", errors.getvalue())

    def test_each_process_initializes_runtime_once(self):
        learn, evaluate = types.ModuleType("learn"), types.ModuleType("eval")
        learn.effect_estimate, evaluate.evaluation = mock.Mock(), mock.Mock()
        torch, mp = types.ModuleType("torch"), types.ModuleType("torch.multiprocessing")
        torch.__path__ = []
        torch.multiprocessing = mp
        mp.set_start_method = mock.Mock()
        launcher._load_runtime.cache_clear()
        self.addCleanup(launcher._load_runtime.cache_clear)
        with mock.patch.dict(sys.modules, {"learn": learn, "eval": evaluate, "torch": torch, "torch.multiprocessing": mp}):
            first, second = launcher._load_runtime(), launcher._load_runtime()
        self.assertIs(first, second)
        self.assertEqual(first, (learn.effect_estimate, evaluate.evaluation))
        mp.set_start_method.assert_called_once_with("spawn", force=True)

    def test_real_spawn_worker_loads_runtime_and_processes_tasks(self):
        import importlib
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        runner = importlib.import_module("run_experiment")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            temporary = Path(directory)
            for filename, function, event in (
                ("learn.py", "effect_estimate", "train"), ("eval.py", "evaluation", "evaluate")
            ):
                source = "def {}(args):\n    with open(args['events'], 'a') as output:\n        output.write({!r})\n".format(
                    function, event + "\n")
                (temporary / filename).write_text(source, encoding="utf-8")
            torch = temporary / "torch"
            torch.mkdir()
            (torch / "__init__.py").write_text("", encoding="utf-8")
            (torch / "multiprocessing.py").write_text("from multiprocessing import set_start_method\n", encoding="utf-8")
            events = temporary / "events.txt"
            entry = {"model_name": "ppicfr", "events": str(events)}
            with mock.patch.object(sys, "path", [str(temporary)] + sys.path):
                with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
                    for _ in range(2):
                        self.assertIsNone(pool.submit(runner.process_model, entry).result(timeout=30))
            self.assertEqual(events.read_text(), "train\nevaluate\ntrain\nevaluate\n")


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for relative in ("configs/default.yaml", "configs/IHDP/default.yaml", "configs/IHDP/ppicfr.yaml"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative, encoding="utf-8")
        self.configs = {
            "configs/default.yaml": {
                "model_name": "unused", "device": "cpu", "n_experiments": 50, "epochs": 1000,
                "batch_size": 128, "verbose": 1,
            },
            "configs/IHDP/default.yaml": {"data_path": "data/example.npz", "pred_output_dir": "results/reference"},
            "configs/IHDP/ppicfr.yaml": {"epochs": 100},
        }

    def write_list(self, entries):
        path = self.root / "experiments.json"
        path.write_text(json.dumps(entries), encoding="utf-8")
        return path

    def entry(self, **extra):
        return dict(model_name="ppicfr", dataset="IHDP", config="configs/IHDP/ppicfr.yaml", **extra)

    def check(self, entries, check_data=False):
        path = self.write_list(entries)
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        issues = []
        checker.check_experiments(self.root, path, issues, check_data, yaml_stub(self.configs))
        after = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        return issues

    def test_merge_precedence_keeps_runtime_name_and_settings(self):
        issues = self.check([self.entry(params={"epochs": 2, "model_name": "ignored"})])
        message = next(i["message"] for i in issues if i["code"] == "settings")
        self.assertIn("model=ppicfr", message)
        self.assertIn("epochs=2", message)
        self.assertIn("repetitions=50", message)
        self.assertFalse(any(i["level"] == "error" for i in issues))

    def test_missing_data_is_warning_until_explicitly_required(self):
        issues = self.check([self.entry()])
        self.assertEqual(next(i["level"] for i in issues if i["code"] == "missing_data"), "warning")
        issues = self.check([self.entry()], check_data=True)
        self.assertEqual(next(i["level"] for i in issues if i["code"] == "missing_data"), "error")

    def test_npz_is_only_checked_for_existence(self):
        data = self.root / "data/example.npz"
        data.parent.mkdir()
        data.write_text("not even an NPZ; preflight must not deserialize it", encoding="utf-8")
        issues = self.check([self.entry()], check_data=True)
        self.assertTrue(any(i["code"] == "data_file" for i in issues))
        self.assertFalse(any(i["code"] == "missing_data" for i in issues))

    def test_invalid_entries_are_rejected_and_ppicfr_is_accepted(self):
        for entries in ({"model_name": "ppicfr"}, [], [None], [self.entry(params=[])],
                        [{"model_name": "ppicfr", "dataset": "IHDP"}]):
            with self.subTest(entries=entries):
                self.assertTrue(any(i["level"] == "error" for i in self.check(entries)))
        self.assertFalse(any(i["level"] == "error" for i in self.check([self.entry()])))

    def test_missing_baseline_dependency_only_blocks_selected_branch(self):
        with mock.patch.object(checker, "package_status", return_value=(False, None)):
            self.assertFalse(any(i["code"] == "baseline_dependency" for i in self.check([self.entry()])))
            issues = self.check([self.entry(params={"teacher_model_name": "escfr"})])
        self.assertTrue(any(i["code"] == "baseline_dependency" for i in issues))

    def test_mixed_case_baseline_and_crf_import_dependencies(self):
        for name in ("DRlearner", "crf"):
            with self.subTest(name=name), mock.patch.object(
                checker, "package_status", side_effect=lambda distribution, module: (module != "xgboost", None)
            ):
                entry = self.entry()
                entry["model_name"] = name
                issues = self.check([entry])
            self.assertTrue(any(i["code"] == "baseline_dependency" and "xgboost" in i["message"] for i in issues))

    def test_default_output_omits_optional_package_details(self):
        def packages(issues, generators):
            checker.add(issues, "info", "optional_package", "optional details")
            checker.add(issues, "warning", "test", "actionable warning")
        with mock.patch.object(checker, "check_packages", side_effect=packages), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(checker.main([]), 0)
        self.assertNotIn("optional details", out.getvalue())
        self.assertIn("actionable warning", out.getvalue())

    def test_missing_yaml_is_reported_not_replaced(self):
        (self.root / "configs/IHDP/default.yaml").unlink()
        issues = self.check([self.entry()])
        self.assertTrue(any(i["code"] == "missing_config" for i in issues))

    def test_case_difference_is_detected_even_on_case_insensitive_filesystems(self):
        directory = self.root / "configs/Synthetic"
        directory.mkdir()
        (directory / "default.yaml").write_text("placeholder", encoding="utf-8")
        issues = []
        checker.check_config_path(self.root, self.root / "configs/synthetic/default.yaml", issues)
        self.assertTrue(any(i["code"] == "path_case" for i in issues))

    def test_cache_and_shared_directory_are_reported_without_removal(self):
        output = self.root / "results/reference"
        output.mkdir(parents=True)
        for name in ("eval_result.txt", "eval_result_summary.txt"):
            (output / name).write_text("retained", encoding="utf-8")
        issues = self.check([self.entry(), self.entry()])
        self.assertTrue(any(i["code"] == "cached_results" for i in issues))
        self.assertTrue(any(i["code"] == "shared_output" for i in issues))

    def test_missing_yaml_dependency_leaves_explicit_unchecked_warning(self):
        path = self.write_list([self.entry()])
        issues = []
        checker.check_experiments(self.root, path, issues, yaml_module=None)
        self.assertTrue(any(i["code"] == "yaml_unavailable" for i in issues))
        self.assertFalse(any(i["code"] == "settings" for i in issues))

    def test_numpy_compatibility_is_only_a_blocker_in_generator_mode(self):
        for generators, level in ((False, "warning"), (True, "error")):
            issues = []
            with mock.patch.object(checker, "package_status", return_value=(True, "2.0.0")):
                checker.check_packages(issues, generators)
            self.assertEqual(next(i["level"] for i in issues if i["code"] == "numpy_generator_compatibility"), level)

    def test_json_report_and_exit_codes(self):
        with mock.patch.object(checker, "check_packages") as packages, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(checker.main(["--json"]), 0)
        packages.assert_called_once()
        self.assertEqual(json.loads(out.getvalue())["project"], "PPICFR")
        with mock.patch.object(checker, "check_packages", side_effect=lambda issues, generators:
                               checker.add(issues, "error", "test", "blocker")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(checker.main(["--json"]), 1)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            checker.main(["--check-data"])
        self.assertEqual(raised.exception.code, 2)

    def test_inaccessible_config_directory_is_reported(self):
        issues = []
        with mock.patch.object(Path, "iterdir", side_effect=PermissionError("no directory listing")):
            checker.check_config_path(self.root, self.root / "configs/default.yaml", issues)
        self.assertTrue(any(i["code"] == "path_access" for i in issues))

    def test_unresolvable_output_is_reported_without_writes(self):
        path = self.write_list([self.entry()])
        before = path.read_bytes()
        for error in (OSError("invalid output"), RuntimeError("symlink loop"), ValueError("invalid path")):
            issues = []
            with mock.patch.object(Path, "resolve", side_effect=error):
                checker.check_experiments(self.root, path, issues, yaml_module=yaml_stub(self.configs))
            self.assertTrue(any(i["code"] == "output_path" for i in issues))
        self.assertEqual(path.read_bytes(), before)

    def test_path_failure_keeps_json_report_parseable(self):
        with mock.patch.object(checker, "check_packages"), \
                mock.patch.object(checker, "check_experiments", side_effect=PermissionError("blocked path")), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(checker.main(["--experiments", "examples/model_ihdp.json", "--json"]), 1)
        report = json.loads(out.getvalue())
        self.assertEqual(report["errors"], 1)
        self.assertEqual(report["issues"][-1]["code"], "path_access")

    def test_bad_json_is_a_reported_error(self):
        path = self.root / "bad.json"
        path.write_text("{invalid", encoding="utf-8")
        issues = []
        checker.check_experiments(self.root, path, issues)
        self.assertEqual(issues[0]["code"], "invalid_experiments")


class ProjectTests(unittest.TestCase):
    def test_model_modules_define_learn_imports(self):
        modules = {}
        for path in ("model/layers.py", "model/ppicfr.py", "model/baselines.py"):
            tree = ast.parse((ROOT / path).read_text())
            modules[path] = {node.name for node in tree.body
                             if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        self.assertTrue({"ShareNetwork", "PrpsyNetwork", "Mu1Network", "Mu0Network",
                         "TauNetwork", "TransformerNetwork"}.issubset(modules["model/layers.py"]))
        self.assertTrue({"PPICFR", "PPICFR_on_DRCFR"}.issubset(modules["model/ppicfr.py"]))
        self.assertIn("DRCFR", modules["model/baselines.py"])
        learning = ast.parse((ROOT / "learn.py").read_text())
        for node in ast.walk(learning):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("model."):
                if node.module == "model.dataset" or node.module == "model.util":
                    continue
                module_path = node.module.replace(".", "/") + ".py"
                self.assertIn(module_path, modules, node.module)
                for alias in node.names:
                    self.assertIn(alias.name, modules[module_path], alias.name)
        calls = {node.func.id for node in ast.walk(learning) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)}
        self.assertTrue({"PPICFR", "PPICFR_on_DRCFR"}.issubset(calls))

    def test_model_module_cross_references_resolve(self):
        defined = set()
        for path in ("model/layers.py", "model/ppicfr.py", "model/baselines.py"):
            tree = ast.parse((ROOT / path).read_text())
            defined |= {node.name for node in tree.body
                        if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        for path in ("model/ppicfr.py", "model/baselines.py"):
            tree = ast.parse((ROOT / path).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "model.layers":
                    for alias in node.names:
                        self.assertIn(alias.name, defined, alias.name)

    def test_example_configuration_files_and_portability_checks(self):
        for path in (ROOT / "examples").glob("*.json"):
            for entry in json.loads(path.read_text()):
                self.assertEqual(entry["model_name"], "ppicfr")
                for relative in (entry["config"], checker.DATASET_CONFIGS[entry["dataset"]]):
                    issues = []
                    configured = ROOT / relative
                    exists = checker.check_config_path(ROOT, configured, issues)
                    current = ROOT
                    for part in Path(relative).parts:
                        matches = [child for child in current.iterdir() if child.name.casefold() == part.casefold()]
                        self.assertEqual(len(matches), 1)
                        current = matches[0]
                    self.assertTrue(current.is_file())
                    if current != configured:
                        self.assertTrue(any(i["code"] == "path_case" for i in issues))
                    else:
                        self.assertTrue(exists)
                        self.assertFalse(issues)

    def test_documentation_links_resolve(self):
        import re
        for path in [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "docs/USAGE.md"]:
            for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
                if not link.startswith(("https://", "http://", "#")):
                    self.assertTrue((path.parent / link.split("#")[0]).exists(), (path, link))


if __name__ == "__main__":
    unittest.main()
