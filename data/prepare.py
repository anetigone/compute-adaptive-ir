"""Prepare pinned image manifests without executing remote dataset scripts.

Default: only DIV2K_train_HR PNGs from dwchoi/div2k800-ffhq1000, split by
clean source image into 640/80/80 for engineering validation. These are NOT the
original DIV2K validation/test split or CBSD68/Urban100 research benchmarks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any

from .denoising import build_datasets, data_config, load_rgb, stable_seed

DEFAULT_HF_REPO = "dwchoi/div2k800-ffhq1000"
DEFAULT_HF_PREFIX = "DIV2K_train_HR"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ROLES = ("train", "val", "test")


def _resolve(path: str | Path, project_root: Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else project_root / value).resolve()


def _rank(items: list[dict], seed: int) -> list[dict]:
    return sorted(items, key=lambda item: (stable_seed(seed, "source-split", item["source_id"]), item["source_id"]))


def _split_pool(items: list[dict], config: dict) -> list[dict]:
    if len(items) < 3:
        raise ValueError("At least three distinct source images are required for train/val/test")
    ratios = [float(x) for x in config.get("split_ratios", [0.8, 0.1, 0.1])]
    if len(ratios) != 3 or min(ratios) <= 0 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("split_ratios must be three positive values summing to one")
    ordered = _rank(items, int(config.get("seed", 20261023)))
    train_count = min(len(items) - 2, max(1, round(len(items) * ratios[0])))
    val_count = min(len(items) - train_count - 1, max(1, round(len(items) * ratios[1])))
    for i, item in enumerate(ordered):
        item["role"] = "train" if i < train_count else "val" if i < train_count + val_count else "test"
    return ordered


def _limit(items: list[dict], config: dict) -> list[dict]:
    maximum = config.get("max_images")
    if maximum is None:
        return items
    result = []
    for role in ROLES:
        limit = maximum.get(role) if isinstance(maximum, dict) else maximum
        selected = [x for x in items if x["role"] == role]
        selected = _rank(selected, int(config.get("seed", 20261023)))
        if limit is not None:
            if isinstance(limit, bool) or int(limit) < 1:
                raise ValueError("max_images must be positive, or a role-to-positive-count mapping")
            selected = selected[:int(limit)]
        result.extend(selected)
    return result


def _folder_items(config: dict, project_root: Path) -> tuple[list[dict], dict]:
    value = config.get("local_dir")
    if not value:
        raise ValueError("source=folder requires data.local_dir")
    folder = _resolve(value, project_root)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    dataset_id = config.get("dataset_id", "local-folder")

    def collect(directory: Path, source_split: str, role: str | None = None) -> list[dict]:
        found = []
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                relative = path.relative_to(folder).as_posix()
                item = {"path": str(path.resolve()), "source_id": relative, "source_split": source_split,
                        "image_id": f"{dataset_id}:{relative}"}
                if role is not None:
                    item["role"] = role
                found.append(item)
        return found

    if all((folder / role).is_dir() for role in ROLES):
        items = [item for role in ROLES for item in collect(folder / role, role, role)]
        split_method = "explicit-role-directories"
    elif (folder / "train").is_dir() and (folder / "validation").is_dir():
        items = collect(folder / "train", "train", "train")
        development = _rank(collect(folder / "validation", "validation"), int(config.get("seed", 20261023)))
        if len(development) < 2:
            raise ValueError("A validation folder needs at least two images for separate val/test roles")
        default_val = max(1, round(0.4 * len(development)))
        val_count = min(len(development) - 1, max(1, int(config.get("val_images", default_val))))
        for i, item in enumerate(development):
            item["role"] = "val" if i < val_count else "test"
        items.extend(development)
        split_method = "original-train-preserved-validation-split-into-val-test"
    else:
        # A partial role layout is likely a misspelling; do not silently mix it.
        if any((folder / role).is_dir() for role in (*ROLES, "validation")):
            raise ValueError("Folder must contain train/val/test, train/validation, or only a flat/ordinary image tree")
        items = _split_pool(collect(folder, "pool"), config)
        split_method = "source-hash-pool-split"
    metadata = {"dataset_id": dataset_id, "local_dir": str(folder), "revision": None,
                "split_method": split_method, "available_role_counts": dict(Counter(x["role"] for x in items)),
                "license": "User-supplied local images; original source terms apply"}
    return _limit(items, config), metadata


def _hf_items(config: dict, cache_root: Path) -> tuple[list[dict], dict]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise ImportError("Hugging Face source requires huggingface_hub; folder source remains offline") from exc
    repo_id = config.get("hf_repo", DEFAULT_HF_REPO)
    prefix = config.get("hf_prefix", DEFAULT_HF_PREFIX).strip("/")
    revision = config.get("revision", "main")
    endpoint = config.get("hf_endpoint") or os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    timeout = float(config.get("hf_timeout", 20))
    # Read metadata only; never import or execute code from the dataset repo.
    info = HfApi(endpoint=endpoint).dataset_info(repo_id, revision=revision, timeout=timeout)
    resolved_revision = info.sha
    if not resolved_revision:
        raise RuntimeError("HF dataset_info did not return a commit SHA; cannot pin downloads")
    all_items = []
    for sibling in info.siblings or []:
        relative = sibling.rfilename
        if relative.startswith(prefix + "/") and Path(relative).suffix.lower() == ".png":
            all_items.append({"source_id": relative, "source_split": "train", "hf_path": relative,
                              "image_id": f"{repo_id}:{relative}"})
    expected = config.get("expected_images", 800 if repo_id == DEFAULT_HF_REPO and prefix == DEFAULT_HF_PREFIX else None)
    if expected is not None and len(all_items) != int(expected):
        raise ValueError(f"Expected {expected} clean HR files under {prefix}, found {len(all_items)} at {resolved_revision}")
    all_items = _split_pool(all_items, config)
    counts = dict(Counter(x["role"] for x in all_items))
    items = _limit(all_items, config)
    repository_dir = cache_root / "hub_files" / repo_id.replace("/", "--") / resolved_revision
    for position, item in enumerate(items, 1):
        print(f"Preparing HF clean image {position}/{len(items)}: {item['source_id']}", flush=True)
        item["path"] = str(Path(hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename=item["hf_path"], revision=resolved_revision,
            local_dir=repository_dir, cache_dir=cache_root / "hub_cache", endpoint=endpoint,
            etag_timeout=timeout,
        )).resolve())
        item["hf_repo"] = repo_id
        item["hf_revision"] = resolved_revision
    card = getattr(info, "card_data", None)
    license_value = card.get("license") if card is not None and hasattr(card, "get") else None
    metadata = {
        "dataset_id": repo_id, "requested_revision": revision, "revision": resolved_revision,
        "hf_prefix": prefix, "endpoint": endpoint, "available_role_counts": counts,
        "downloaded_images": len(items), "split_method": "source-hash-pool-split",
        "hf_dataset_card_license": license_value,
        "original_dataset": "DIV2K" if repo_id == DEFAULT_HF_REPO else None,
        "original_source_url": "https://data.vision.ee.ethz.ch/cvl/DIV2K/" if repo_id == DEFAULT_HF_REPO else None,
        "source_card_url": f"https://huggingface.co/datasets/{repo_id}/blob/{resolved_revision}/README.md",
        "license": "DIV2K: academic research use; original image copyrights remain with their owners" if repo_id == DEFAULT_HF_REPO else "Refer to dataset source terms",
        "protocol_note": "Only source train HR images are used; derived train/val/test engineering roles are not original DIV2K validation/test or CBSD68/Urban100",
    }
    return items, metadata


def _image_metadata(item: dict) -> dict:
    path = Path(item["path"])
    file_hash = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            file_hash.update(block)
    rgb = load_rgb(path)
    height, width = rgb.shape[:2]
    pixel_hash = hashlib.sha256(struct.pack(">II", width, height) + rgb.tobytes()).hexdigest()
    return {**item, "path": str(path.resolve()), "sha256": file_hash.hexdigest(), "pixel_sha256": pixel_hash,
            "width": width, "height": height, "channels": 3, "bytes": path.stat().st_size}


def prepare_data(config: dict, root: Path) -> Path:
    """Return a content-addressed manifest; root is the project root directory."""
    cfg = data_config(config)
    project_root = Path(root).resolve()
    cache_root = _resolve(cfg.get("root", "data_cache"), project_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    source = cfg.get("source", "huggingface")
    if source == "huggingface":
        items, metadata = _hf_items(cfg, cache_root)
    elif source in {"folder", "local"}:
        items, metadata = _folder_items(cfg, project_root)
    else:
        raise ValueError("data.source must be huggingface or folder; synthetic images are only test fixtures")
    records = [_image_metadata(item) for item in items]
    counts = Counter(x["role"] for x in records)
    if any(counts[role] == 0 for role in ROLES):
        raise ValueError(f"Every role needs at least one clean source image, got {dict(counts)}")
    identities: dict[str, dict] = {}
    for record in records:
        previous = identities.get(record["pixel_sha256"])
        if previous is not None:
            raise ValueError(f"Duplicate clean source pixels detected: {previous['source_id']} ({previous['role']}) and {record['source_id']} ({record['role']}); remove duplicate images before splitting")
        identities[record["pixel_sha256"]] = record
    manifest = {
        "schema_version": 1, "source": source, "source_metadata": metadata, "config": cfg,
        "role_counts": dict(counts), "subset": cfg.get("max_images") is not None,
        "duplicate_check": {"method": "exact normalized RGB pixel SHA256", "scope": "selected_source_images_only",
                            "selected_image_count": len(records), "not_checked": "unselected/undownloaded sources and perceptual near-duplicates"},
        "records": sorted(records, key=lambda x: (ROLES.index(x["role"]), x["image_id"])),
        "preprocessing": {"color": "RGB", "exif_orientation": "applied", "noise_protocol": cfg.get("noise_protocol", "clip_uint8"),
                          "noise_scale": "sigma in 0--255 units", "train_patch": cfg.get("train_patch", 128),
                          "eval_mode": cfg.get("eval_mode", "center_crop"), "eval_crop": cfg.get("eval_crop", 256)},
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    manifest_path = cache_root / f"manifest_{digest[:16]}.json"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != text:
        raise RuntimeError("Manifest filename collision; refusing to overwrite")
    if not manifest_path.exists():
        manifest_path.write_text(text, encoding="utf-8")
    # Fail early on invalid crop sizes/protocols; no images or network touched.
    build_datasets(cfg, manifest_path)
    print(f"Prepared {dict(counts)} clean source images. Manifest: {manifest_path}", flush=True)
    return manifest_path


def _read_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("YAML configs require PyYAML; JSON configs use only the standard library") from exc
    return yaml.safe_load(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root, not the cache directory")
    parser.add_argument("--max-images", type=int, help="Download/select at most this many sources per final role")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.max_images is not None:
        if "data" in config:
            config["data"]["max_images"] = args.max_images
        else:
            config["max_images"] = args.max_images
    prepare_data(config, args.root)


if __name__ == "__main__":
    main()
