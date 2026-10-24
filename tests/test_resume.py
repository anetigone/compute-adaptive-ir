"""An interrupted optimization must follow the same trajectory as a full run."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from common import load_checkpoint
from data import prepare_data
from tests.support import make_local_config
from train import train


class ResumeContractTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        self.previous_determinism = torch.are_deterministic_algorithms_enabled()

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)
        torch.use_deterministic_algorithms(self.previous_determinism)

    def assert_nested_equal(self, first, second, path="checkpoint"):
        if isinstance(first, torch.Tensor):
            self.assertIsInstance(second, torch.Tensor, path)
            torch.testing.assert_close(first, second, rtol=0, atol=0, msg=path)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second), path)
            for key in first:
                self.assert_nested_equal(first[key], second[key], f"{path}.{key}")
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second), path)
            for index, (left, right) in enumerate(zip(first, second)):
                self.assert_nested_equal(left, right, f"{path}[{index}]")
        else:
            self.assertEqual(first, second, path)

    def test_resume_preserves_weights_optimizer_schedule_rng_and_data_position(self):
        previous_threads = torch.get_num_threads()
        previous_determinism = torch.are_deterministic_algorithms_enabled()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = make_local_config(root)
                manifest = prepare_data(config, root)
                with redirect_stdout(io.StringIO()):
                    continuous_path = train(config, manifest, root / "continuous")
                    partial_path = train(config, manifest, root / "resumed", stop_after=2)
                partial = load_checkpoint(partial_path)
                self.assertEqual(partial["step"], 2)
                self.assertEqual(partial["scheduler"]["last_epoch"], 2)
                self.assertEqual(partial["data_position"], {"epoch": 1, "batch_offset": 1})
                self.assertEqual(partial["config"]["train"]["max_steps"], 4,
                                 "An interruption must not shorten the planned LR schedule")
                with redirect_stdout(io.StringIO()):
                    resumed_path = train(config, manifest, root / "resumed", resume=partial_path)
                continuous, resumed = load_checkpoint(continuous_path), load_checkpoint(resumed_path)
                self.assertEqual(resumed["step"], 4)
                self.assertEqual(resumed["scheduler"]["last_epoch"], 4)
                self.assertEqual(resumed["data_position"], {"epoch": 2, "batch_offset": 2})
                for key in ("model", "optimizer", "scheduler", "scaler", "rng", "data_position"):
                    with self.subTest(state=key):
                        self.assert_nested_equal(continuous[key], resumed[key], key)
                modified = copy.deepcopy(config)
                modified["train"]["max_steps"] = 5
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    train(modified, manifest, root / "resumed", resume=resumed_path)
        finally:
            torch.set_num_threads(previous_threads)
            torch.use_deterministic_algorithms(previous_determinism)

    def test_off_schedule_interruption_does_not_change_best_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = make_local_config(root)
            manifest = prepare_data(config, root)
            train(config, manifest, root / "continuous")
            resumed_dir = root / "resumed"
            partial_path = train(config, manifest, resumed_dir, stop_after=1)
            self.assertFalse((resumed_dir / "best.pt").exists())
            summary = json.loads((resumed_dir / "training_summary.json").read_text(encoding="utf-8"))
            self.assertIsNone(summary["best_validation_psnr"])
            first_row = json.loads((resumed_dir / "train.jsonl").read_text(encoding="utf-8"))
            self.assertNotIn("validation", first_row)
            train(config, manifest, resumed_dir, resume=partial_path)
            continuous_best = load_checkpoint(root / "continuous" / "best.pt")
            resumed_best = load_checkpoint(resumed_dir / "best.pt")
            self.assertEqual(continuous_best["step"], resumed_best["step"])
            self.assertEqual(continuous_best["best_psnr"], resumed_best["best_psnr"])
            self.assert_nested_equal(continuous_best["model"], resumed_best["model"])
            for run in (root / "continuous", resumed_dir):
                rows = [json.loads(line) for line in (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual([row["step"] for row in rows if "validation" in row], [2, 4])

    def test_resume_rejects_changes_to_validation_selection_protocol(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = make_local_config(root)
            manifest = prepare_data(config, root)
            run_dir = root / "run"
            checkpoint = train(config, manifest, run_dir, stop_after=1)
            for key, changed_value in (("val_every", 1), ("val_max_samples", 2)):
                changed = copy.deepcopy(config)
                changed["train"][key] = changed_value
                with self.subTest(setting=key), self.assertRaisesRegex(ValueError, "configuration changed"):
                    train(changed, manifest, run_dir, resume=checkpoint)

    def test_resume_backs_up_and_removes_uncommitted_log_rows_and_broken_tail(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = make_local_config(root)
            manifest = prepare_data(config, root)
            run_dir = root / "run"
            checkpoint = train(config, manifest, run_dir, stop_after=1)
            log = run_dir / "train.jsonl"
            crashed_log = (log.read_text(encoding="utf-8") +
                           json.dumps({"step": 2, "stale_before_resume": True}) + "\n" +
                           '{"step": 3, "broken":')
            log.write_text(crashed_log, encoding="utf-8")
            train(config, manifest, run_dir, resume=checkpoint, stop_after=2)
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["step"] for row in rows], [1, 2])
            self.assertTrue(all("stale_before_resume" not in row for row in rows))
            backups = list(run_dir.glob("train.before_resume_*.jsonl"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), crashed_log)

    def test_resume_cannot_mix_a_checkpoint_into_another_run_directory(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = make_local_config(root)
            manifest = prepare_data(config, root)
            checkpoint = train(config, manifest, root / "original", stop_after=1)
            destination = root / "other_run"
            with self.assertRaisesRegex(ValueError, "original run directory"):
                train(config, manifest, destination, resume=checkpoint)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
