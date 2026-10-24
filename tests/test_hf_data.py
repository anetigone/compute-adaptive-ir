"""HF transport is mocked: test revision pinning and source selection offline."""

import copy
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from data import prepare_data
from tests.support import make_local_config


class HuggingFacePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = make_local_config(self.root)
        self.config["data"].update(source="huggingface", revision="main",
                                   hf_endpoint="https://offline-test.invalid")
        self.config["data"].pop("local_dir")
        self.commit = "a" * 40

    def tearDown(self):
        self.temporary.cleanup()

    def mock_hub(self, image_count):
        files = [f"DIV2K_train_HR/{index:04d}.png" for index in range(1, image_count + 1)]
        files += ["FFHQ/0001.png", "DIV2K_valid_HR/0001.png",
                  "DIV2K_train_HR/readme.txt", "DIV2K_train_HR/not_png.jpg"]
        info = types.SimpleNamespace(
            sha=self.commit, siblings=[types.SimpleNamespace(rfilename=name) for name in reversed(files)],
            card_data={"license": "test-fixture"},
        )
        api = Mock()
        api.dataset_info.return_value = info
        api_factory = Mock(return_value=api)

        def download(**kwargs):
            # Materialize only the files the production code actually requests.
            filename = kwargs["filename"]
            path = self.root / "fake_remote_files" / filename
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                pixels = np.random.default_rng(int(path.stem)).integers(0, 256, (32, 40, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(path)
            return str(path)

        downloader = Mock(side_effect=download)
        module = types.ModuleType("huggingface_hub")
        module.HfApi = api_factory
        module.hf_hub_download = downloader
        return module, api_factory, api, downloader

    def test_800_source_roles_are_assigned_before_subset_and_downloads_pin_commit(self):
        module, api_factory, api, downloader = self.mock_hub(800)
        self.config["data"]["max_images"] = 1
        with patch.dict(sys.modules, {"huggingface_hub": module}), redirect_stdout(io.StringIO()):
            path = prepare_data(self.config, self.root)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_metadata"]["available_role_counts"],
                         {"train": 640, "val": 80, "test": 80})
        self.assertEqual(manifest["role_counts"], {"train": 1, "val": 1, "test": 1})
        self.assertEqual(manifest["source_metadata"]["revision"], self.commit)
        self.assertEqual(len(manifest["records"]), 3)
        self.assertEqual(downloader.call_count, 3, "A smoke subset must not download all 800 sources")
        api_factory.assert_called_once_with(endpoint="https://offline-test.invalid")
        api.dataset_info.assert_called_once_with("dwchoi/div2k800-ffhq1000", revision="main", timeout=20.0)
        for call in downloader.call_args_list:
            arguments = call.kwargs
            self.assertEqual(arguments["revision"], self.commit)
            self.assertEqual(arguments["repo_type"], "dataset")
            self.assertEqual(arguments["repo_id"], "dwchoi/div2k800-ffhq1000")
            self.assertTrue(arguments["filename"].startswith("DIV2K_train_HR/"))
            self.assertTrue(arguments["filename"].endswith(".png"))
            self.assertIn(self.commit, str(arguments["local_dir"]))

    def test_source_role_does_not_change_when_selecting_a_small_subset(self):
        module, _factory, _api, downloader = self.mock_hub(20)
        self.config["data"]["expected_images"] = 20
        with patch.dict(sys.modules, {"huggingface_hub": module}), redirect_stdout(io.StringIO()):
            full_path = prepare_data(self.config, self.root)
            full = json.loads(full_path.read_text(encoding="utf-8"))
            full_roles = {record["source_id"]: record["role"] for record in full["records"]}
            self.assertEqual(downloader.call_count, 20)
            downloader.reset_mock()
            subset_config = copy.deepcopy(self.config)
            subset_config["data"]["max_images"] = 1
            subset_path = prepare_data(subset_config, self.root)
        subset = json.loads(subset_path.read_text(encoding="utf-8"))
        self.assertEqual(downloader.call_count, 3)
        for record in subset["records"]:
            self.assertEqual(record["role"], full_roles[record["source_id"]])


if __name__ == "__main__":
    unittest.main()
