"""Helper utilities for loading bundled third-party dependencies."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


def ensure_torch_two_sample() -> None:
    """Expose the vendored ``torch_two_sample`` package if it's bundled.

    The packaged project includes the upstream ``torch-two-sample`` sources under
    ``distribution_shift_perception/torch-two-sample``. When the wheel is
    installed, setuptools places those files inside the package but does not add
    that directory to ``sys.path``. If the external dependency is not installed,
    we prepend the vendored location so ``import torch_two_sample`` works.
    """

    try:
        importlib.import_module("torch_two_sample")
        return
    except ModuleNotFoundError:
        pass

    vendor_root = Path(__file__).resolve().parent / "torch-two-sample"
    candidate = vendor_root / "torch_two_sample"
    if candidate.is_dir():
        sys.path.insert(0, str(vendor_root))
        try:
            importlib.import_module("torch_two_sample")
        except ModuleNotFoundError as exc:  # pragma: no cover - diagnostics only
            raise ModuleNotFoundError(
                "Vendored torch_two_sample package missing compiled extension"
            ) from exc


def bootstrap() -> None:
    ensure_torch_two_sample()

``