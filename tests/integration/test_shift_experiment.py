import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import torch

from distribution_shift_perception.data.data_builder import get_dataloader
from distribution_shift_perception.models.autoencoder import ConvAutoencoderFC
from distribution_shift_perception.shift_experiment import ShiftExperiment


class DatasetIntegrationTests(unittest.TestCase):
    """Integration tests that exercise the real datasets and pipeline."""

    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).resolve().parents[2]
        cls.source_dir = project_root / "datasets" / "CULane"
        cls.target_dir = project_root / "datasets" / "Curvelanes"
        cls.source_list = cls.source_dir / "list" / "train.txt"
        cls.target_list = cls.target_dir / "train" / "train.txt"

        missing = [
            str(path)
            for path in (cls.source_dir, cls.target_dir, cls.source_list, cls.target_list)
            if not path.exists()
        ]
        if missing:
            raise unittest.SkipTest(
                "Skipping integration tests; required dataset files missing: "
                + ", ".join(missing)
            )

        torch.set_num_threads(1)

    def test_get_dataloader_returns_expected_sample_count(self):
        loader, image_paths = get_dataloader(
            root_dir=str(self.source_dir),
            list_path=str(self.source_list),
            batch_size=2,
            image_size=512,
            num_samples=2,
            cropImg=False,
            block_idx=0,
        )

        self.assertEqual(len(image_paths), 2)

        batch = next(iter(loader))
        self.assertEqual(batch.shape, (2, 3, 512, 512))
        self.assertFalse(batch.isnan().any())

    def test_shift_experiment_calibration_and_logging(self):
        class NoPretrainedAutoencoder(ConvAutoencoderFC):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("pretrained", False)
                super().__init__(*args, **kwargs)

        with TemporaryDirectory() as tmpdir, mock.patch(
            "distribution_shift_perception.shift_experiment.ConvAutoencoderFC",
            NoPretrainedAutoencoder,
        ):
            experiment = ShiftExperiment(
                source_dir=str(self.source_dir),
                target_dir=str(self.target_dir),
                source_list_path=str(self.source_list),
                target_list_path=str(self.target_list),
                src_samples=4,
                tgt_samples=3,
                num_runs=1,
                block_idx=0,
                batch_size=2,
                image_size=512,
                num_calib=3,
                alpha=0.1,
                seed_base=123,
                shift=None,
                std=0.0,
                cropImg=False,
                file_location=tmpdir,
                file_name="integration_log.json",
                save_all_image_paths=True,
            )

            experiment.run()

            self.assertTrue(np.isfinite(experiment.tau))
            self.assertEqual(len(experiment.null_stats), experiment.num_calib)

            log_path = Path(tmpdir) / "integration_log.json"
            self.assertTrue(log_path.exists())

            with log_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            self.assertIn("experiments", payload)
            self.assertTrue(payload["experiments"])
            experiment_log = payload["experiments"][0]

            calibration = experiment_log["data"].get("Calibration", {})
            self.assertIn("Result", calibration)
            self.assertGreaterEqual(calibration["Result"]["Tau"], 0.0)

            sanity = experiment_log["data"].get("Sanity Check", {})
            self.assertIn("Shift Detected", sanity)
            self.assertFalse(sanity["Shift Detected"])

            data_shift = experiment_log["data"].get("Data Shift Test Data", {})
            self.assertIn("TPR", data_shift)
            self.assertIn("Average MMD", data_shift)


if __name__ == "__main__":
    unittest.main()
