"""Check behavior that must survive changing the restoration architecture."""

import unittest

import torch
from torch.nn import functional as F

from models import build_model


def tiny_model(name="reusable_refiner", steps=4):
    return build_model({
        "name": name,
        "width": 4,
        "block_counts": [1, 1, 1, 1, 1],
        "steps": steps,
    })


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(731)

    def test_shared_trajectory_matches_fixed_exits_and_extrapolation(self):
        model = tiny_model().eval()
        image = torch.rand(1, 3, 13, 17)
        parameters_before = tuple(id(value) for value in model.parameters())
        with torch.no_grad():
            trajectory = model.forward_trajectory(image, steps=6)
            for step in (1, 2, 4, 6):
                with self.subTest(step=step):
                    fixed = model(image, steps=step)
                    self.assertEqual(fixed.shape, image.shape)
                    torch.testing.assert_close(fixed, trajectory[step - 1], rtol=0, atol=0)
        self.assertEqual(parameters_before, tuple(id(value) for value in model.parameters()))

    def test_fixed_path_caches_observation_and_decodes_only_at_exit(self):
        model = tiny_model().eval()
        calls = {"stem": 0, "update": 0, "readout": 0, "gain_head": 0}

        def count(name):
            def hook(*_args):
                calls[name] += 1
            return hook

        handles = [
            model.stem0.register_forward_hook(count("stem")),
            model.update.register_forward_hook(count("update")),
            model.readout.register_forward_hook(count("readout")),
            model.gain_head.register_forward_hook(count("gain_head")),
        ]
        try:
            with torch.no_grad():
                model(torch.rand(1, 3, 12, 16), steps=4)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(calls, {"stem": 1, "update": 4, "readout": 1, "gain_head": 0})

    def test_late_exit_backpropagates_through_earlier_shared_states(self):
        model = tiny_model().train()
        states = []

        def keep_state(_module, _inputs, output):
            output.retain_grad()
            states.append(output)

        handle = model.update.register_forward_hook(keep_state)
        try:
            outputs = model.forward_trajectory(torch.rand(1, 3, 12, 16), steps=3)
            F.l1_loss(outputs[-1], torch.rand_like(outputs[-1])).backward()
        finally:
            handle.remove()
        self.assertEqual(len(states), 3)
        self.assertIsNotNone(states[0].grad, "Detaching recurrent state silently truncates training")
        self.assertGreater(states[0].grad.abs().sum().item(), 0)
        for name in ("stem0.weight", "update.delta.weight", "readout.weight"):
            with self.subTest(parameter=name):
                gradient = dict(model.named_parameters())[name].grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all().item())
                self.assertGreater(gradient.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
        # Zero-initialized NAF residual scales legitimately give some branch
        # weights zero gradient initially, so do not require every gradient > 0.
        self.assertTrue(all(not parameter.requires_grad for parameter in model.gain_head.parameters()))

    def test_dynamic_exit_really_skips_unneeded_updates(self):
        model = tiny_model().eval()
        image = torch.rand(1, 3, 13, 17)
        calls = []
        executions = {"update": 0, "readout": 0}

        def count(name):
            def hook(*_args):
                executions[name] += 1
            return hook

        def costs(step, available, height, width):
            calls.append((step, available, height, width))
            return [float(value + 1) for value in range(available)]

        handles = [model.update.register_forward_hook(count("update")),
                   model.readout.register_forward_hook(count("readout"))]
        try:
            with torch.no_grad():
                first = model.forward_dynamic(image, lambda_db_per_ms=float("inf"), extra_cost_ms=costs)
                self.assertEqual(executions, {"update": 1, "readout": 1})
                torch.testing.assert_close(first.image, model(image, steps=1), rtol=0, atol=0)
                self.assertEqual(first.steps, 1)
                self.assertEqual(calls, [(1, 3, 13, 17)])
                calls.clear()
                executions.update(update=0, readout=0)
                last = model.forward_dynamic(image, lambda_db_per_ms=-float("inf"), extra_cost_ms=costs)
                self.assertEqual(executions, {"update": 4, "readout": 1})
                torch.testing.assert_close(last.image, model(image, steps=4), rtol=0, atol=0)
                self.assertEqual(last.steps, 4)
                self.assertEqual(calls, [(1, 3, 13, 17), (2, 2, 13, 17), (3, 1, 13, 17)])
        finally:
            for handle in handles:
                handle.remove()

    def test_fixed_baseline_has_one_exit_and_preserves_tiny_image_shape(self):
        baseline = tiny_model("nafnet", steps=1).eval()
        with torch.no_grad():
            image = torch.rand(1, 3, 1, 3)
            output = baseline(image)
            self.assertEqual(output.shape, image.shape)
            self.assertTrue(torch.isfinite(output).all().item())
            torch.testing.assert_close(output, baseline.forward_trajectory(image)[0], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            baseline(image, steps=2)


if __name__ == "__main__":
    unittest.main()
