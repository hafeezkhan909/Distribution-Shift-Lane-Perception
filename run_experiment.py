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

    def __init__(self, root_dir, split="train", image_size=512, dataShift=None, cropImage=False):
        self.shift = dataShift
        self.cropImage = cropImage
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
            img = apply_shift(img, self.shift)
        if self.cropImage:
            w, h = img.size
            img = img.crop((0, h//2, w, h)) # left, top, right, bottom
            return self.transform(img)
        else:
            return self.transform(img)


# =========================================================
# Dataloader helpers
# =========================================================
def get_dataloader(
    dataset_name, split, batch_size, image_size, num_samples, block_idx=0, cropImage=False
):
    root = f"datasets/{dataset_name}"
    ds = LaneImageDataset(root, split, image_size, dataShift=None, cropImage=False)
    start, end = block_idx * num_samples, min((block_idx + 1) * num_samples, len(ds))
    subset = Subset(ds, list(range(start, end)))
    print(f"[INFO] {dataset_name} ({split}) → [{start}:{end}] ({len(subset)} samples)")
    return DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )


def get_seeded_random_dataloader(
    dataset_name, split, batch_size, image_size, num_samples, seed, shift=None, cropImage=False
):
    root = f"datasets/{dataset_name}"
    ds = LaneImageDataset(root, split, image_size, dataShift=shift, cropImage=False)
    random.seed(seed)
    chosen = random.sample(range(len(ds)), min(num_samples, len(ds)))
    subset = Subset(ds, chosen)
    # print(
    #     f"[INFO] {dataset_name} ({split}) → Random {len(chosen)} samples (seed={seed})"
    # )
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
        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)
            z = model.encode(imgs)
            if z.dim() > 2:
                raise ValueError("Images are still in the pixel space")
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
    num_runs: int = 10,
    block_idx: int = 0, #block of samples selected from the the text file
    batch_size: int = 16, #batch processing of data within an epoch
    image_size: int = 512,
    num_calib: int = 100,
    alpha: float = 0.05,
    seed_base: int = 42,
    shift: str = None,
    std: float = 0.0):

    print(f"CUDA Avalible: {torch.cuda.is_available()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("features", exist_ok=True)

    # ------------------ Model ------------------
    print("\nInitializing autoencoder...")
    model = ConvAutoencoderFC(latent_dim=512, pretrained=True).to(device)

    # ------------------ Source ------------------
    src_loader = get_dataloader(
        source, src_split, batch_size, image_size, src_samples, block_idx
    )
    src_feats = extract_features(model, src_loader, device)
    print(f"{source} features loaded successfully !")

    # ------------------ Calibration ------------------
    print("\n[STEP 1] Calibration: same-domain")
    null_stats = []
    for i in trange(num_calib, desc="Calibrating"):
        seed = seed_base + i
        calib_src_loader = get_seeded_random_dataloader(
            source, src_split, batch_size, image_size, tgt_samples, seed, shift=None
        )
        calib_src_feats = extract_features(model, calib_src_loader, device)
        t_stat = mmd_test(src_feats, calib_src_feats)
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
    mmd_val = mmd_test(src_feats, sanity_src_feats)
    print(f"[CHECK] MMD({source}→{source}) = {mmd_val:.6f}, τ = {tau:.6f}")
    print(
        "No shift detected (expected same-domain match)."
        if mmd_val <= tau
        else "Unexpected shift."
    )

    shift_object = None
    if shift == "gaussian":
        if std == 0.0:
            raise ValueError("Gaussian noise selected but std=0. Please provide > 0 --std value.")
        shift_object = GaussianShift(std=std)

    # =========================================================
    # NEW SECTION: Data Shift test (source → target)
    # =========================================================
    print(f"\n[STEP] Data Shift test: {source} → {target} using same τ")

    tpr_list = []
    mmd_values = []

    for run in trange(num_runs, desc="Shift Testing"):
        seed_cross = seed_base + run
        tgt_loader_cross = get_seeded_random_dataloader(
            target,
            tgt_split,
            batch_size,
            image_size,
            tgt_samples,
            seed_cross,
            shift=shift_object,
        )
        tgt_feats_cross = extract_features(model, tgt_loader_cross, device)
        mmd_cross = mmd_test(src_feats, tgt_feats_cross)
        mmd_values.append(mmd_cross)
        detected = mmd_cross > tau
        tpr_list.append(int(detected))

        print(f"[RUN {run+1:03d}] MMD={mmd_cross:.6f} {'✅' if detected else '❌'}")

    # ---- Summarize results ----
    tpr = np.mean(tpr_list)
    print("\n[RESULTS] Data Shift detection summary")
    print(f"Average MMD: {np.mean(mmd_values):.6f} ± {np.std(mmd_values):.6f}")
    print(f"TPR (true positive rate) over {num_runs} runs: {tpr*100:.2f}%")
    print(f"Shifted-Data test: {shift_object}")
    np.save(f"features/mmd_{source}_{target}_100runs.npy", np.array(mmd_values))
    np.save(f"features/tpr_{source}_{target}_100runs.npy", np.array(tpr_list))
    np.save(f"features/tau_{source}_{target}.npy", np.array([tau]))


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
        "-g", "--tgt_split", type=str, default="valid", help="Target dataset split"
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
        "--num_runs",
        type=int,
        default=10,
        help="Number of runs",
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
    parser.add_argument("--shift", type=str, default=None,
                        help="Type of shift: gaussian | rotate | translate etc.")
    parser.add_argument("--std", type=float, default=0.0,
                        help="Std for Gaussian noise shift")
    
    args = parser.parse_args()
    
    main(**vars(args))
