"""AMP overflows must not corrupt weights or consume scheduler steps."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from common import load_checkpoint
from data import prepare_data
from models import build_model
from tests.support import make_local_config
from train import optimizer_step, train


class AmpStepTests(unittest.TestCase):
    def test_inf_and_nan_are_skipped_and_scale_backs_off(self):
        for value in (float("inf"), float("nan")):
            with self.subTest(gradient=value):
                parameter = torch.nn.Parameter(torch.tensor([1.0]))
                optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.0)
                scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
                scaler.scale(parameter.square().sum()).backward()
                parameter.grad.fill_(value)
                self.assertIsNone(optimizer_step(optimizer, scaler, [parameter], 1.0))
                self.assertEqual(parameter.item(), 1.0)
                self.assertFalse(optimizer.state)
                self.assertEqual(scaler.get_scale(), 4.0)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(parameter.square().sum()).backward()
                self.assertEqual(optimizer_step(optimizer, scaler, [parameter], 1.0), 2.0)
                self.assertLess(parameter.item(), 1.0)

    def test_nonfinite_without_amp_still_raises(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([float("inf")])
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        with self.assertRaises(FloatingPointError):
            optimizer_step(optimizer, torch.amp.GradScaler("cpu", enabled=False), [parameter], 1.0)
        self.assertEqual(parameter.item(), 1.0)

    def test_finite_large_gradients_do_not_overflow_norm_calculation(self):
        parameter = torch.nn.Parameter(torch.zeros(4))
        parameter.grad = torch.full_like(parameter, 1e30)
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        norm = optimizer_step(optimizer, torch.amp.GradScaler("cpu", enabled=False), [parameter], 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(norm, dtype=torch.float64)).item())
        torch.testing.assert_close(parameter, torch.full_like(parameter, -0.05))

    def test_training_only_counts_successful_optimizer_updates(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = make_local_config(root)
            config["model"] = {"name": "nafnet", "width": 4, "block_counts": [1, 1, 1, 1, 1], "steps": 1}
            config["train"].update(max_steps=2, grad_accumulation=1, exit_weights=None)
            manifest = prepare_data(config, root)
            model = build_model(config["model"])
            calls = 0

            def overflow_once(gradient):
                nonlocal calls
                calls += 1
                return torch.full_like(gradient, float("inf")) if calls == 1 else gradient

            next(model.parameters()).register_hook(overflow_once)
            scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
            with patch("train.build_model", return_value=model), patch("train.torch.amp.GradScaler", return_value=scaler):
                checkpoint = train(config, manifest, root / "run")
            saved = load_checkpoint(checkpoint)
            self.assertEqual(saved["step"], 2)
            self.assertEqual(saved["scheduler"]["last_epoch"], 2)
            self.assertEqual(saved["amp_skipped_updates"], 1)
            self.assertEqual(calls, 3)
            self.assertEqual(saved["scaler"]["scale"], 4.0)
            for state in saved["optimizer"]["state"].values():
                self.assertEqual(state["step"].item(), 2)
            rows = [json.loads(line) for line in (root / "run" / "train.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["event"], "amp_overflow")
            self.assertEqual(rows[0]["step"], 0)
            self.assertEqual([row["step"] for row in rows if "loss_l1" in row], [1, 2])


if __name__ == "__main__":
    unittest.main()
