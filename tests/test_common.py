"""RNG snapshots must round-trip through safe checkpoint serialization."""

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from common import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint


class CheckpointUtilityTests(unittest.TestCase):
    def test_safe_checkpoint_restores_python_numpy_and_torch_rng(self):
        # This contract must also work on a machine with no CUDA installation.
        with patch("torch.cuda.is_available", return_value=False):
            original = capture_rng_state()
            try:
                random.seed(161)
                np.random.seed(162)
                torch.manual_seed(163)
                state = capture_rng_state()
                expected = (random.random(), np.random.rand(4), torch.rand(4))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "checkpoint.pt"
                    save_checkpoint(path, {"step": 2, "rng": state, "tensor": torch.arange(3)})
                    self.assertFalse(path.with_suffix(".pt.tmp").exists())
                    loaded = load_checkpoint(path, device="cpu")
                self.assertEqual(loaded["step"], 2)
                restore_rng_state(loaded["rng"])
                self.assertEqual(random.random(), expected[0])
                np.testing.assert_array_equal(np.random.rand(4), expected[1])
                torch.testing.assert_close(torch.rand(4), expected[2], rtol=0, atol=0)
            finally:
                restore_rng_state(original)


if __name__ == "__main__":
    unittest.main()
