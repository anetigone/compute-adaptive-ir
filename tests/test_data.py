"""Data identities, splitting and stochastic inputs are part of the protocol."""

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

    def test_manifest_source_roles_and_pixel_content_do_not_overlap(self):
        path = self.prepare()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        records = manifest["records"]
        self.assertEqual({role: sum(r["role"] == role for r in records)
                          for role in ("train", "val", "test")},
                         {"train": 3, "val": 2, "test": 2})
        for identity in ("image_id", "source_id", "pixel_sha256"):
            roles = {}
            for record in records:
                roles.setdefault(record[identity], set()).add(record["role"])
            self.assertTrue(all(len(value) == 1 for value in roles.values()), identity)
        # Preparation must not depend on filesystem enumeration order.
        again = json.loads(self.prepare().read_text(encoding="utf-8"))
        self.assertEqual(records, again["records"])

    def test_duplicate_pixels_under_another_filename_cannot_cross_roles(self):
        source = Path(self.config["data"]["local_dir"])
        # Re-encoding changes file bytes while preserving source pixels.
        with Image.open(source / "train" / "image_0.png") as image:
            image.save(source / "val" / "duplicate.png", compress_level=0)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_changed_source_file_is_rejected_before_training_or_evaluation(self):
        path = self.prepare()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        image_path = Path(manifest["records"][0]["path"])
        with Image.open(image_path) as image:
            # Different encoding suffices: manifest file hashes protect the
            # exact immutable inputs even if normalized pixels happen to match.
            image.save(image_path.with_name("replacement.png"), compress_level=0)
        image_path.write_bytes(image_path.with_name("replacement.png").read_bytes())
        with self.assertRaises(ValueError):
            build_datasets(self.config, path)

    def test_forged_manifest_cannot_reuse_a_source_identity_across_roles(self):
        path = self.prepare()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        train = next(record for record in manifest["records"] if record["role"] == "train")
        val = next(record for record in manifest["records"] if record["role"] == "val")
        val["source_id"] = train["source_id"]
        forged = self.root / "forged_manifest.json"
        forged.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            build_datasets(self.config, forged)

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
