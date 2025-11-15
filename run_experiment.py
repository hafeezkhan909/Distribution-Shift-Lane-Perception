import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from autoencoder import ConvAutoencoderFC
import argparse

# Import the new, specific shift classes from data_utils
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


# =========================================================
# Dataset
# =========================================================
class LaneImageDataset(Dataset):
    """Generic dataset for lane images given a root path and list file."""

    def __init__(self, root_dir, split="train", image_size=512, dataShift=None):
        self.shift = dataShift
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        # list file logic: Needs modularization for any data loading.
        if "Curvelanes" in root_dir:
            list_path = os.path.join(root_dir, split, f"{split}.txt") # for Curvelanes txt file extraction
        else:
            list_path = os.path.join(root_dir, "list", f"{split}.txt") # for CULane txt file extraction 

        if not os.path.exists(list_path):
            raise FileNotFoundError(f"List file not found: {list_path}")

        with open(list_path, "r") as f:
            self.image_paths = [line.strip() for line in f.readlines() if line.strip()]

        self.transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx].lstrip("/")
        if "Curvelanes" in self.root_dir:
            img_path = os.path.join(self.root_dir, self.split, rel_path)
        else:
            img_path = os.path.join(self.root_dir, rel_path)

        img = Image.open(img_path).convert("RGB")
        if self.shift is not None:
            img_shifted = apply_shift(img, self.shift)
            return self.transform(img_shifted)
        else:
            return self.transform(img)


# =========================================================
# Dataloader helpers
# =========================================================
def get_dataloader(
    dataset_name, split, batch_size, image_size, num_samples, block_idx=0
):
    root = f"datasets/{dataset_name}"
    ds = LaneImageDataset(root, split, image_size, dataShift=None)
    start, end = block_idx * num_samples, min((block_idx + 1) * num_samples, len(ds))
    subset = Subset(ds, list(range(start, end)))
    print(f"[INFO] {dataset_name} ({split}) → [{start}:{end}] ({len(subset)} samples)")
    return DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )


def get_seeded_random_dataloader(
    dataset_name, split, batch_size, image_size, num_samples, seed, shift=None
):
    root = f"datasets/{dataset_name}"
    ds = LaneImageDataset(root, split, image_size, dataShift=shift)
    random.seed(seed)
    chosen = random.sample(range(len(ds)), min(num_samples, len(ds)))
    subset = Subset(ds, chosen)
    print(
        f"[INFO] {dataset_name} ({split}) → Random {len(chosen)} samples (seed={seed})"
    )
    return DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )


# =========================================================
# Feature extraction
# =========================================================
def extract_features(model, loader, device):
    model.eval()
    feats = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(device, non_blocking=True)
            z = model.encode(imgs)
            if z.dim() > 2:
                z = z.view(z.size(0), -1) # code to run on raw images (to flatten the image and do the tests)

            feats.append(z.cpu().numpy())
    return np.concatenate(feats, axis=0)


# =========================================================
# Combined pipeline
# =========================================================
def main(
    source: str = "CULane",
    target: str = "Curvelanes",
    src_split: str = "train", # train or test or val split
    tgt_split: str = "test",
    src_samples: int = 1000, # No. of source samples as train set passed
    tgt_samples: int = 100,
    block_idx: int = 0, #block of samples selected from the the text file
    batch_size: int = 16, #batch processing of data within an epoch
    image_size: int = 512,
    num_calib: int = 100,
    alpha: float = 0.05,
    seed_base: int = 42):

    print(f"CUDA Avalible: {torch.cuda.is_available()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("features", exist_ok=True)

    # ------------------ Model ------------------
    print("\nInitializing autoencoder...")
    model = ConvAutoencoderFC(latent_dim=512, pretrained=True).to(device)

    # ------------------ Source ------------------
    src_path = f"features/{source}_{src_split}_{src_samples}_{block_idx}.npy"
    if os.path.exists(src_path):
        print(f"Src feats already exist, loading from path: {src_path}")
        src_feats = np.load(src_path)
    else:
        src_loader = get_dataloader(
            source, src_split, batch_size, image_size, src_samples, block_idx
        )
        src_feats = extract_features(model, src_loader, device)
        np.save(src_path, src_feats)
        print(f"[SAVED] {src_path} ({src_feats.shape})")

    # ------------------ Calibration ------------------
    print("\n[STEP 1] Calibration: same-domain")
    null_stats = []
    for i in trange(num_calib, desc="Calibrating"):
        seed = seed_base + i
        calib_src_loader = get_seeded_random_dataloader(
            source, src_split, batch_size, image_size, tgt_samples, seed, shift=None
        )
        calib_src_feats = extract_features(model, calib_src_loader, device)
        t_stat, _ = mmd_test(src_feats, calib_src_feats)
        null_stats.append(t_stat)

    null_stats = np.array(null_stats)
    tau = np.percentile(null_stats, 100 * (1 - alpha))
    print(f"\n[RESULT] τ({1 - alpha:.2f}) = {tau:.6f}")
    print(
        f"[RESULT] Mean MMD (no shift): {null_stats.mean():.6f} ± {null_stats.std():.6f}"
    )
    np.save("features/calibration_null_mmd.npy", null_stats)

    # ------------------ Sanity Check ------------------
    print(f"\n[STEP 2] Sanity Check: {source}→{source}")
    seed_match = seed_base + 1
    sanity_src_loader = get_seeded_random_dataloader(
        source, src_split, batch_size, image_size, tgt_samples, seed_match, shift=None
    )
    sanity_src_feats = extract_features(model, sanity_src_loader, device)
    mmd_val, _ = mmd_test(src_feats, sanity_src_feats)
    print(f"[CHECK] MMD({source}→{source}) = {mmd_val:.6f}, τ = {tau:.6f}")
    print(
        "No shift detected (expected same-domain match)."
        if mmd_val <= tau
        else "Unexpected shift."
    )

    # =========================================================
    # NEW SECTION: Cross-domain test (CULane → Curvelanes)
    # =========================================================
    print("\n[STEP] Cross-domain test: CULane → Curvelanes using same τ")
    target_cross = target

    # ---- Run 1 random seeds ----
    num_runs = 1
    tpr_list = []
    mmd_values = []

    for run in trange(num_runs, desc="Cross-domain seeds"):
        seed_cross = seed_base + 100 + run  # avoid overlap with calibration seeds
        tgt_loader_cross = get_seeded_random_dataloader(
            target_cross,
            tgt_split,
            batch_size,
            image_size,
            tgt_samples,
            seed_cross,
            shift=None,
        )
        tgt_feats_cross = extract_features(model, tgt_loader_cross, device)
        mmd_cross, _ = mmd_test(src_feats, tgt_feats_cross)
        mmd_values.append(mmd_cross)
        detected = mmd_cross > tau
        tpr_list.append(int(detected))

        print(f"[RUN {run+1:03d}] MMD={mmd_cross:.6f} {'✅' if detected else '❌'}")

    # ---- Summarize results ----
    tpr = np.mean(tpr_list)
    print("\n[RESULTS] Cross-domain detection summary")
    print(f"Average MMD: {np.mean(mmd_values):.6f} ± {np.std(mmd_values):.6f}")
    print(f"TPR (true positive rate) over {num_runs} runs: {tpr*100:.2f}%")
    np.save("features/mmd_curvelanes_100runs.npy", np.array(mmd_values))
    np.save("features/tpr_curvelanes_100runs.npy", np.array(tpr_list))

    # =========================================================
    # NEW SECTION: Shifted-Data test (CULane → Shifted CULane)
    # =========================================================
    print("\n[STEP] Shifted-Data test: CULane → Shifted CULane using same τ")
    target_cross = target

    # ---- Run 200 random seeds per shift ----
    num_runs = 200

    # Use the new DataShift subclasses from data_utils.py
    shifts_list = [
        GaussianShift(std=1),
        GaussianShift(std=10),
        GaussianShift(std=20),
        GaussianShift(std=30),
        GaussianShift(std=40),
        GaussianShift(std=50),
        GaussianShift(std=60),
        GaussianShift(std=70),
        GaussianShift(std=80),
        GaussianShift(std=90),
        GaussianShift(std=100),
    ]

    print("Text for Sanity: With Noise")

    for shift_object in shifts_list:
        print(f"\n[STEP] Shifted-Data test: {shift_object}")
        tpr_list = []
        mmd_values = []

        for run in trange(num_runs, desc="Random CULane seeds"):
            seed_cross = seed_base + 100 + run  # avoid overlap with calibration seeds

            # Pass the entire shift_object to the dataloader
            tgt_loader_cross = get_seeded_random_dataloader(
                target_cross,
                tgt_split,
                batch_size,
                image_size,
                tgt_samples,
                seed_cross,
                shift=shift_object,
            )
            tgt_feats_cross = extract_features(model, tgt_loader_cross, device)
            mmd_cross, _ = mmd_test(src_feats, tgt_feats_cross)
            mmd_values.append(mmd_cross)
            detected = mmd_cross > tau
            tpr_list.append(int(detected))

            # This print is noisy, you might want to comment it out
            print(
                f"[RUN {run+1:03d}] MMD={mmd_cross:.6f} {'✅ Shift Detected' if detected else '❌ Shift not Detected'}"
            )

        # ---- Summarize results ----
        tpr = np.mean(tpr_list)
        print("\n[RESULTS] Shifted-Data detection summary")
        print(f"    Shifted-Data test: {shift_object}")
        print(f"    Average MMD: {np.mean(mmd_values):.6f} ± {np.std(mmd_values):.6f}")
        print(f"    TPR (true positive rate) over {num_runs} runs: {tpr*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Distribution Shift Lane Perception Experiment",
        description="Run MMD-based distribution shift tests on lane datasets.",
    )

    # --- Dataset Arguments ---
    parser.add_argument(
        "-s", "--source", type=str, default="CULane", help="Source dataset name"
    )
    parser.add_argument(
        "-t", "--target", type=str, default="Curvelanes", help="Target dataset name"
    )
    parser.add_argument(
        "-p", "--src_split", type=str, default="train", help="Source dataset split"
    )
    parser.add_argument(
        "-g", "--tgt_split", type=str, default="train", help="Target dataset split"
    )

    # --- Sampling Arguments ---
    parser.add_argument(
        "-r",
        "--src_samples",
        type=int,
        default=1000,
        help="Number of samples for the source reference",
    )
    parser.add_argument(
        "-a",
        "--tgt_samples",
        type=int,
        default=100,
        help="Number of samples for the target test",
    )
    parser.add_argument(
        "-b",
        "--block_idx",
        type=int,
        default=0,
        help="Block index for chunked source loading",
    )

    # --- Model & MMD Test Arguments ---
    parser.add_argument(
        "-i", "--batch_size", type=int, default=16, help="Batch size for feature extraction"
    )
    parser.add_argument(
        "-z", "--image_size", type=int, default=512, help="Image resize dimension"
    )
    parser.add_argument(
        "-e",
        "--num_calib",
        type=int,
        default=100,
        help="Number of calibration runs for null distribution",
    )
    parser.add_argument(
        "-n", "--alpha", type=float, default=0.05, help="Significance level for the test"
    )
    
    # --- Reproducibility ---
    parser.add_argument(
        "--seed_base",
        type=int,
        default=42,
        help="Base seed for random sampling",
    )

    args = parser.parse_args()
    
    # Call main by unpacking the args dictionary.
    # This automatically maps 'args.source' to the 'source' param, etc.
    main(**vars(args))
