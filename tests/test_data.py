"""Dataset splits and repeatable denoising inputs."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from data import build_datasets, prepare_data
from tests.support import make_local_config


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = make_local_config(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self):
        return prepare_data(self.config, self.root)

    def test_file_lists_are_disjoint_and_repeatable(self):
        path = self.prepare()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual({role: len(paths) for role, paths in manifest.items()},
                         {"train": 3, "val": 2, "test": 2})
        paths = [path for group in manifest.values() for path in group]
        self.assertEqual(len(paths), len(set(paths)))
        again = json.loads(self.prepare().read_text(encoding="utf-8"))
        self.assertEqual(manifest, again)

    def test_flat_images_are_split_before_applying_sample_limit(self):
        folder = self.root / "pool"
        folder.mkdir()
        for index in range(20):
            Image.new("RGB", (32, 32), (index, 0, 0)).save(folder / f"{index:02d}.png")
        self.config["data"]["local_dir"] = str(folder)
        full = json.loads(self.prepare().read_text(encoding="utf-8"))
        self.assertEqual({role: len(paths) for role, paths in full.items()}, {"train": 16, "val": 2, "test": 2})
        self.assertEqual(full, json.loads(self.prepare().read_text(encoding="utf-8")))
        self.config["data"]["max_images"] = 1
        subset = json.loads(self.prepare().read_text(encoding="utf-8"))
        for role in full:
            self.assertEqual(subset[role], full[role][:1])

    def test_training_augmentation_is_replayable_but_changes_across_epochs(self):
        manifest = self.prepare()
        first = build_datasets(self.config, manifest)["train"]
        second = build_datasets(self.config, manifest)["train"]
        first.set_epoch(3)
        second.set_epoch(3)
        original = first[0]
        for other in (first[0], second[0]):
            self.assertEqual(original["sample_seed"], other["sample_seed"])
            self.assertEqual(original["noise_seed"], other["noise_seed"])
            torch.testing.assert_close(original["clean"], other["clean"], rtol=0, atol=0)
            torch.testing.assert_close(original["noisy"], other["noisy"], rtol=0, atol=0)
        first.set_epoch(4)
        changed = first[0]
        self.assertNotEqual(original["sample_seed"], changed["sample_seed"])
        self.assertFalse(torch.equal(original["noisy"], changed["noisy"]))
        self.assertFalse(torch.equal(original["clean"], changed["clean"]),
                         "Training must also vary its crop/augmentation, not only its noise")
        first.set_epoch(3)
        torch.testing.assert_close(original["noisy"], first[0]["noisy"], rtol=0, atol=0)

    def test_test_noise_is_independent_of_method_sigma_order_and_access_order(self):
        manifest = self.prepare()
        original = build_datasets(self.config, manifest)["test"]
        expected = {(original[index]["image_id"], original[index]["sigma"]): original[index]
                    for index in range(len(original))}
        changed_config = copy.deepcopy(self.config)
        changed_config["model"]["name"] = "nafnet"
        changed_config["data"]["eval_sigmas"] = [50, 15, 25]
        reordered = build_datasets(changed_config, manifest)["test"]
        for index in reversed(range(len(reordered))):
            sample = reordered[index]
            other = expected[(sample["image_id"], sample["sigma"])]
            self.assertEqual(sample["noise_seed"], other["noise_seed"])
            torch.testing.assert_close(sample["noisy"], other["noisy"], rtol=0, atol=0)
            torch.testing.assert_close(sample["clean"], other["clean"], rtol=0, atol=0)
            self.assertGreaterEqual(sample["noisy"].min().item(), 0)
            self.assertLessEqual(sample["noisy"].max().item(), 1)
            scaled = sample["noisy"] * 255
            self.assertLess((scaled - scaled.round()).abs().max().item(), 2e-5)


if __name__ == "__main__":
    unittest.main()
