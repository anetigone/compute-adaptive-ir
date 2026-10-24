"""Check download selection and the resulting splits without network access."""

import io
import json
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from data import build_datasets, prepare_data
from tests.support import make_local_config


class HuggingFacePreparationTests(unittest.TestCase):
    def test_download_hr_images_then_split_and_load_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_local_config(root)
            config["data"].update(source="huggingface", max_images=1)

            def download(**kwargs):
                folder = Path(kwargs["local_dir"]) / "DIV2K_train_HR"
                folder.mkdir(parents=True, exist_ok=True)
                for index in range(20):
                    Image.new("RGB", (32, 32), (index, 0, 0)).save(folder / f"{index:04d}.png")

            module = types.ModuleType("huggingface_hub")
            module.snapshot_download = Mock(side_effect=download)
            with patch.dict("sys.modules", {"huggingface_hub": module}), redirect_stdout(io.StringIO()):
                path = prepare_data(config, root)

            module.snapshot_download.assert_called_once_with(
                repo_id="dwchoi/div2k800-ffhq1000", repo_type="dataset",
                allow_patterns=["DIV2K_train_HR/*.png"],
                local_dir=root / "cache" / "images", endpoint=None,
            )
            splits = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual({role: len(paths) for role, paths in splits.items()},
                             {"train": 1, "val": 1, "test": 1})
            # max_images limits the split lists, not the initial download.
            self.assertEqual(len(list((root / "cache" / "images").rglob("*.png"))), 20)
            datasets = build_datasets(config, path)
            self.assertEqual(datasets["train"][0]["noisy"].shape, (3, 16, 16))


if __name__ == "__main__":
    unittest.main()
