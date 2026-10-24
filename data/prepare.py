"""Download clean images, split them, and save train/val/test file lists."""

import json
import random
from pathlib import Path

from .denoising import data_config


def prepare_data(config: dict, root: Path) -> Path:
    cfg = data_config(config)
    cache = (Path(root) / cfg.get("root", "data_cache")).resolve()
    cache.mkdir(parents=True, exist_ok=True)

    if cfg.get("source", "huggingface") == "huggingface":
        from huggingface_hub import snapshot_download

        prefix = cfg.get("hf_prefix", "DIV2K_train_HR")
        snapshot_download(
            repo_id=cfg.get("hf_repo", "dwchoi/div2k800-ffhq1000"),
            repo_type="dataset",
            allow_patterns=[f"{prefix}/*.png"],
            local_dir=cache / "images",
            endpoint=cfg.get("hf_endpoint"),
        )
        image_dir = cache / "images" / prefix
    elif cfg["source"] == "folder":
        image_dir = (Path(root) / cfg["local_dir"]).resolve()
    else:
        raise ValueError("data.source must be huggingface or folder")

    def images(directory):
        return sorted(str(path.resolve()) for path in directory.rglob("*")
                      if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"})

    roles = ("train", "val", "test")
    if all((image_dir / role).is_dir() for role in roles):
        splits = {role: images(image_dir / role) for role in roles}
    else:
        paths = images(image_dir)
        if len(paths) < 3:
            raise ValueError(f"Need at least 3 images to split: {image_dir}")
        ratios = cfg.get("split_ratios", [0.8, 0.1, 0.1])
        if len(ratios) != 3 or min(ratios) <= 0 or abs(sum(ratios) - 1) > 1e-6:
            raise ValueError("split_ratios must be three positive values summing to 1")
        random.Random(cfg.get("seed", 20261023)).shuffle(paths)
        train_end = min(len(paths) - 2, max(1, int(len(paths) * ratios[0])))
        val_end = min(len(paths) - 1, train_end + max(1, int(len(paths) * ratios[1])))
        splits = {"train": paths[:train_end], "val": paths[train_end:val_end], "test": paths[val_end:]}

    maximum = cfg.get("max_images")
    if maximum is not None:
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise ValueError("max_images must be a positive integer")
        splits = {role: paths[:maximum] for role, paths in splits.items()}
    if any(not paths for paths in splits.values()):
        raise ValueError("train/val/test must each contain at least one image")

    split_path = cache / "splits.json"
    split_path.write_text(json.dumps(splits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = {role: len(paths) for role, paths in splits.items()}
    print(f"Prepared {counts}: {split_path}", flush=True)
    return split_path
