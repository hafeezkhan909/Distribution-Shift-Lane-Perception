import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import torch
from autoencoder import ConvAutoencoderFC
import argparse

from data_utils import (
    ShiftTypes,
    apply_shift,
    DataShift,
    GaussianShift,
    RotationShift,
    TranslationShift,
    ShearShift,
    ZoomShift,
    HorizontalFlipShift,
    VerticalFlipShift,
)
from mmd_test import mmd_test
from data_builder import (
    get_dataloader,
    get_seeded_random_dataloader
)

# ---------Feature extraction---------
def extract_features(model, loader, device):
    model.eval()
    feats = []
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)
            z = model.encode(imgs)
            if z.dim() > 2:
                raise ValueError("Images are still in the pixel space")
                z = z.view(z.size(0), -1) # code to run on raw images (to flatten the image and do the tests)

            feats.append(z.cpu().numpy())
    return np.concatenate(feats, axis=0)

class ShiftExperiment:
    def __init__(
            self,
            source: str = "CULane",
            target: str = "Curvelanes",
            src_split: str = "train", # train or test or val split
            tgt_split: str = "test",
            src_samples: int = 1000, # No. of source samples as train set passed
            tgt_samples: int = 100,
            num_runs: int = 10,
            block_idx: int = 0, #block of samples selected from the the text file
            batch_size: int = 16, #batch processing of data within an epoch
            image_size: int = 512,
            num_calib: int = 100,
            alpha: float = 0.05,
            seed_base: int = 42,
            shift: str = None,
            std: float = 0.0,
            cropImg: bool = False
    ):
        
        self.source = source
        self.target = target
        self.src_split = src_split
        self.tgt_split = tgt_split
        self.src_samples = src_samples
        self.tgt_samples = tgt_samples
        self.num_runs = num_runs
        self.block_idx = block_idx
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_calib = num_calib
        self.alpha = alpha
        self.seed_base = seed_base
        self.shift_type = shift
        self.std = std
        self.cropImg = cropImg

        print(f"CUDA Avalible: {torch.cuda.is_available()}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs("features", exist_ok=True)

        # ------------------ Model ------------------
        print("\nInitializing autoencoder...")
        self.model = ConvAutoencoderFC(latent_dim=512, pretrained=True).to(self.device)

        # Initialialize shift object
        self.shift_object = None
        if self.shift_type == "gaussian":
            if self.std == 0.0:
                raise ValueError("Gaussian noise selected but std=0. Please provide > 0 --std value.")
            self.shift_object = GaussianShift(std=self.std)

        
    # STEP 0 — Load Source Features
    def load_source_features(self):
        loader = get_dataloader(
            self.source, self.src_split, self.batch_size, self.image_size,
            self.src_samples, self.cropImg, self.block_idx
        )
        self.src_feats = extract_features(self.model, loader, self.device)
        print(f"{self.source} features loaded. Shape = {self.src_feats.shape}\n")
    
    # STEP 1 — Calibration (Null Distribution)
    def calibrate(self):
        print(f"[STEP 1] Calibration using {self.source}...")
        null_stats = []

        for i in trange(self.num_calib, desc="Calibrating"):
            seed = self.seed_base + i
            calib_src_loader = get_seeded_random_dataloader(
                self.source, self.src_split, self.batch_size, self.image_size,
                self.tgt_samples, seed, self.cropImg, shift=None
            )
            calib_src_feats = extract_features(self.model, calib_src_loader, self.device)

            t_stat = mmd_test(self.src_feats, calib_src_feats)
            null_stats.append(t_stat)

        self.null_stats = np.array(null_stats)
        self.tau = np.percentile(self.null_stats, 100 * (1 - self.alpha))

        print(f"\n[RESULT] τ({1 - self.alpha:.2f}) = {self.tau:.6f}")
        print(f"Mean MMD (same-distribution): {self.null_stats.mean():.6f} ± {self.null_stats.std():.6f}\n")

    # STEP 2 — Sanity Check
    def sanity_check(self):
        print("[STEP 2] Sanity Check...")

        sanity_src_loader = get_seeded_random_dataloader(
            self.source, self.src_split, self.batch_size, self.image_size,
            self.tgt_samples, self.seed_base + 1, self.cropImg, shift=None
        )
        sanity_src_feats = extract_features(self.model, sanity_src_loader, self.device)

        mmd_val = mmd_test(self.src_feats, sanity_src_feats)
        print(f"[SANITY CHECK] MMD({self.source}→{self.source}) = {mmd_val:.6f}, τ = {self.tau:.6f}")

        if mmd_val <= self.tau:
            print("No shift detected.\n")
        else:
            print("False shift detected.\n")

    # STEP 3 — Data Shift Test
    def data_shift_test(self):
        print(f"[STEP 3] Data Shift Test: {self.source} → {self.target}, Noise applied: {self.shift_object}, with std: {self.std}\n")

        tpr_list = []
        mmd_values = []

        for i in trange(self.num_runs, desc="Shift Testing"):
            seed = self.seed_base + i
            tgt_loader_cross = get_seeded_random_dataloader(
                self.target, self.tgt_split, self.batch_size, self.image_size,
                self.tgt_samples, seed, self.cropImg, shift=self.shift_object
            )
            tgt_feats_cross = extract_features(self.model, tgt_loader_cross, self.device)
            mmd_cross = mmd_test(self.src_feats, tgt_feats_cross)

            mmd_values.append(mmd_cross)
            detected = mmd_cross > self.tau
            tpr_list.append(int(detected))

            print(f"[RUN {i+1}] MMD={mmd_cross:.6f} {'✅' if detected else '❌'}")

        tpr_result = np.mean(tpr_list)
        print("\n[RESULTS] Data Shift detection summary")
        print(f"Noise Applied: {self.shift_object}")
        print(f"Average MMD: {np.mean(mmd_values):.6f} ± {np.std(mmd_values):.6f}")
        print(f"TPR (true positive rate) over {self.num_runs} runs: {tpr_result*100:.2f}%")

    # RUN EVERYTHING
    def run(self):
        # Step 0
        self.load_source_features()
        # Step 1
        self.calibrate()
        # Step 2
        self.sanity_check()
        # Step 3
        self.data_shift_test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=str, default="CULane")
    parser.add_argument("--target", type=str, default="Curvelanes")
    parser.add_argument("--src_split", type=str, default="train")
    parser.add_argument("--tgt_split", type=str, default="valid")
    parser.add_argument("--src_samples", type=int, default=1000)
    parser.add_argument("--tgt_samples", type=int, default=100)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--block_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_calib", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--shift", type=str, default=None)
    parser.add_argument("--std", type=float, default=0.0)
    parser.add_argument("--cropImg", type=bool, default=False)

    args = parser.parse_args()

    ShiftExperiment(**vars(args)).run()
