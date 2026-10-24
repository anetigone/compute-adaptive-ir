"""RGB denoising data; this package deliberately does not shadow HF datasets."""

from .denoising import DenoisingDataset, build_datasets, stable_seed


def prepare_data(config, root):
    # Keep `python -m data.prepare` free of a runpy module-import warning.
    from .prepare import prepare_data as prepare
    return prepare(config, root)


__all__ = ["DenoisingDataset", "build_datasets", "prepare_data", "stable_seed"]
