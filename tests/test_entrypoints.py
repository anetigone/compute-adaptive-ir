"""User entrypoints must preserve explicit budgets and previous artifacts."""

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from models import build_model
from run import _model_config, smoke


class EntrypointRegressionTests(unittest.TestCase):
    def test_selecting_the_same_model_preserves_the_explicit_train_steps_alias(self):
        config = {"model": {"name": "reusable_refiner", "width": 4,
                            "block_counts": [1, 1, 1, 1, 1], "train_steps": 2}}
        selected = _model_config(config, "reusable_refiner")
        self.assertEqual(build_model(selected["model"]).default_steps, 2)
        self.assertEqual(selected, config)
        self.assertIsNot(selected, config)
        baseline = _model_config(config, "nafnet")
        self.assertEqual(build_model(baseline["model"]).default_steps, 1)
        self.assertEqual(config["model"]["train_steps"], 2)

    def test_nonempty_smoke_directory_is_rejected_before_rewriting_fixtures(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "outputs"
            fixture = output_root / "smoke" / "fixture" / "train" / "train_00.png"
            fixture.parent.mkdir(parents=True)
            Image.new("RGB", (80, 80), (10, 20, 30)).save(fixture)
            previous = fixture.read_bytes()
            args = types.SimpleNamespace(config=None, device="cpu", name="smoke", full_model=True)
            with patch("run._configuration", return_value={"output_root": str(output_root)}), \
                    patch("run._synthetic_folder") as write_fixture:
                with self.assertRaises(FileExistsError):
                    smoke(args)
                write_fixture.assert_not_called()
            self.assertEqual(fixture.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
