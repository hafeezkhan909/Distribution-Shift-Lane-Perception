import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from autoencoder import ConvAutoencoderFC
import random

# from time import time

# --------------------------------------------------------
# Dataset Loader (path-aware and split-flexible)
# --------------------------------------------------------


class LaneImageDataset(Dataset):
    """
    Generic dataset for lane images given a root path and list file.
    Automatically adjusts paths for CULane and CurveLanes.
    """

    def __init__(self, root_dir, split="train", image_size=512):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        # Build list file path (depends on dataset structure)
        if "Curvelanes" in root_dir:
            list_path = os.path.join(root_dir, split, f"{split}.txt")
        else:
            list_path = os.path.join(root_dir, "list", f"{split}.txt")

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

        # Handle CurveLanes structure (e.g., datasets/Curvelanes/train/images/…)
        if "Curvelanes" in self.root_dir:
            img_path = os.path.join(self.root_dir, self.split, rel_path)
        else:
            img_path = os.path.join(self.root_dir, rel_path)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img


# --------------------------------------------------------
# Utility: Create dataloader with subset option
# --------------------------------------------------------


def get_dataloader(
    dataset_name,
    split="train",
    batch_size=8,
    image_size=512,
    num_samples=100,
    block_idx=0,
):
    """
    Returns dataloader for selected dataset and split.

    Args:
        dataset_name (str): Name of the dataset.
        split (str): Data split ('train', 'test', etc.).
        batch_size (int): Batch size for the dataloader.
        image_size (int): Resize all images to this size.
        num_samples (int): Number of samples per subset (e.g., 1000).
        block_idx (int): Index of the subset block (0 for first 1k, 1 for next 1k, etc.).

    Example:
        block_idx=0 → samples [0:1000]
        block_idx=1 → samples [1000:2000]
        block_idx=2 → samples [2000:3000]
    """
    from feature_extractor import (
        LaneImageDataset,
    )  # local import to avoid circular deps

    root = f"datasets/{dataset_name}"
    ds = LaneImageDataset(root, split=split, image_size=image_size)
    total_len = len(ds)

    # Compute start and end indices for this block
    start_idx = block_idx * num_samples
    end_idx = min(start_idx + num_samples, total_len)

    if start_idx >= total_len:
        raise ValueError(
            f"[ERROR] block_idx={block_idx} exceeds dataset size ({total_len})."
        )

    indices = list(range(start_idx, end_idx))
    ds_subset = Subset(ds, indices)

    loader = DataLoader(
        ds_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    print(
        f"[INFO] {dataset_name} ({split}) → Samples [{start_idx}:{end_idx}] ({len(indices)} total)."
    )

    return loader


def get_random_dataloader(
    dataset_name, split="train", batch_size=8, image_size=512, num_samples=50
):
    """
    Returns dataloader for selected dataset and split,
    sampling a random subset of num_samples each time it’s called.
    """
    from feature_extractor import (
        LaneImageDataset,
    )  # Import here to avoid circular dependency

    root = f"datasets/{dataset_name}"
    ds = LaneImageDataset(root, split=split, image_size=image_size)

    all_indices = list(range(len(ds)))

    # random.seed(int(time.time() * 1e6) % (2**32 - 1))
    chosen_indices = random.sample(all_indices, min(num_samples, len(ds)))

    subset = Subset(ds, chosen_indices)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    print(f"[INFO] {dataset_name} ({split}) → Random {len(chosen_indices)} samples.")
    return loader


# --------------------------------------------------------
# Feature extraction
# --------------------------------------------------------


def extract_features(model, loader, device):
    model.eval()
    feats = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(device, non_blocking=True)
            z = model.encode(imgs)

            # ✅ Handle both (B, 512, 8, 8) and (B, 512)
            if z.dim() > 2:
                z = z.view(z.size(0), -1)

            feats.append(z.cpu().numpy())

    return np.concatenate(feats, axis=0)


def extract_raw_image_features(loader, device):
    """
    Extracts raw image features (flattened pixel tensors) without using the autoencoder.
    Each image is resized and normalized by the dataset transform, then flattened.
    """
    feats = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="Extracting raw image features"):
            # imgs: [B, 3, H, W]
            imgs = imgs.to(device, non_blocking=True)
            # Flatten each image to a single vector (e.g., 3×512×512 → 786432)
            z = imgs.view(imgs.size(0), -1)
            feats.append(z.cpu().numpy())
    return np.concatenate(feats, axis=0)


# --------------------------------------------------------
# Main pipeline
# --------------------------------------------------------


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initializing untrained autoencoder (UAE)...")
    model = ConvAutoencoderFC(latent_dim=512, pretrained=True).to(device)

    print(
        f"Loading datasets:\n  SRC={args.source} ({args.src_split}, {args.src_samples} samples)\n  TGT={args.target} ({args.tgt_split}, {args.tgt_samples} samples)"
    )

    # Extract source features (only once)
    src_loader = get_dataloader(
        args.source,
        args.src_split,
        args.batch_size,
        args.image_size,
        args.src_samples,
        args.block_idx,
    )
    os.makedirs("features", exist_ok=True)
    src_path = f"features/{args.source}_{args.src_split}_{args.src_samples}_{args.block_idx}.npy"

    if os.path.exists(src_path):
        print(f"[SKIP] Found existing source features → {src_path}")
        src_feats = np.load(src_path)
    else:
        src_feats = extract_features(model, src_loader, device)
        np.save(src_path, src_feats)
        print(f"SRC: {src_feats.shape} → saved to {src_path}")

    # Repeat target feature extraction multiple times
    for run_idx in range(args.num_runs):
        print(
            f"\n[RUN {run_idx+1}/{args.num_runs}] Extracting random target features..."
        )
        tgt_loader = get_random_dataloader(
            args.target,
            args.tgt_split,
            args.batch_size,
            args.image_size,
            args.tgt_samples,
        )

        tgt_feats = extract_features(model, tgt_loader, device)
        tgt_path = f"features/{args.target}_{args.tgt_split}_{args.tgt_samples}_run{run_idx}.npy"
        np.save(tgt_path, tgt_feats)
        print(f"[SAVED] {tgt_path}")

    print("All target runs complete.")


# --------------------------------------------------------
# Argument parsing
# --------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract UAE features for source and target datasets."
    )

    parser.add_argument(
        "--source",
        type=str,
        default="CULane",
        help="Source dataset name (e.g., CULane, Curvelanes)",
    )
    parser.add_argument(
        "--target", type=str, default="Curvelanes", help="Target dataset name"
    )
    parser.add_argument(
        "--src_split",
        type=str,
        default="train",
        help="Split for source (train/test/valid)",
    )
    parser.add_argument(
        "--tgt_split",
        type=str,
        default="train",
        help="Split for target (train/test/valid)",
    )
    parser.add_argument(
        "--src_samples", type=int, default=1000, help="Number of source samples"
    )
    parser.add_argument(
        "--block_idx",
        type=int,
        default=0,
        help="subset of samples for source selection",
    )
    parser.add_argument(
        "--tgt_samples", type=int, default=50, help="Number of target samples"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for dataloaders"
    )
    parser.add_argument(
        "--image_size", type=int, default=512, help="Resize images to this size"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of random target runs to generate.",
    )
    args = parser.parse_args()
    main(args)
