"""Minimal sanity check using distribution_shift_perception.

This script loads a handful of samples from the on-disk CULane/Curvelanes
splits, runs the calibration/sanity/data-shift stages, and prints the key
outputs. It is intentionally tiny so you can verify an installation without
waiting for the full experiment setup.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from distribution_shift_perception import ShiftExperiment


def resolve_dataset_paths(project_root: Path) -> tuple[Path, Path, Path, Path]:
    datasets_root = project_root / "datasets"
    source_dir = datasets_root / "CULane"
    target_dir = datasets_root / "Curvelanes"
    source_list = source_dir / "list" / "train.txt"
    target_list = target_dir / "train" / "train.txt"

    missing = [
        path
        for path in (source_dir, target_dir, source_list, target_list)
        if not path.exists()
    ]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Expected datasets not found. Make sure the CULane and Curvelanes "
            f"downloads are available at: {missing_str}"
        )

    return source_dir, target_dir, source_list, target_list


def run_quick_check() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source_dir, target_dir, source_list, target_list = resolve_dataset_paths(
        project_root
    )

    with TemporaryDirectory() as tmpdir:
        experiment = ShiftExperiment(
            source_dir=str(source_dir),
            target_dir=str(target_dir),
            source_list_path=str(source_list),
            target_list_path=str(target_list),
            src_samples=8,
            tgt_samples=4,
            num_runs=1,
            block_idx=0,
            batch_size=2,
            image_size=256,
            num_calib=5,
            alpha=0.1,
            seed_base=2025,
            shift=None,
            std=0.0,
            cropImg=False,
            file_location=tmpdir,
            file_name="quick_sanity.json",
            save_all_image_paths=True,
        )
        experiment.run()

        sanity = experiment.loggerExperimentalData.get("Sanity Check", {})
        if sanity.get("Shift Detected"):
            raise SystemExit("Sanity check failed: detected a shift against itself")

        log_path = Path(tmpdir) / "quick_sanity.json"
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        latest = payload["experiments"][-1]
        calibration = latest["data"]["Calibration"]["Result"]
        data_shift = latest["data"]["Data Shift Test Data"]

        print("Tau (self-calibration):", calibration["Tau"])
        print("Sanity check MMD <= tau:", sanity["Results"]["MMD"] <= calibration["Tau"])
        print("Data-shift TPR (%):", data_shift["TPR"])
        print("Average MMD (target vs source):", data_shift["Average MMD"])
        print("Example target image paths:")
        for path in data_shift["Individual Test Data"][0].get("Image Paths", []):
            print(" •", path)

        assert np.isfinite(experiment.tau), "Tau should be finite for valid data"


if __name__ == "__main__":
    run_quick_check()
