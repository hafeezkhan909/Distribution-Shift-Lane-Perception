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
            std: float = 0.0
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
        
