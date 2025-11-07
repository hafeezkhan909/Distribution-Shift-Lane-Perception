import os
import argparse
import numpy as np
from tqdm import trange
import torch

from mmd_test import mmd_test
from autoencoder import ConvAutoencoderFC
from feature_extractor import (
    get_dataloader as get_src_dataloader,
    get_random_dataloader,
    extract_features,
)

# --------------------------------------------------------
# Calibration: Estimate empirical MMD threshold
# --------------------------------------------------------

def calibrate_threshold(
    dataset_name="CULane",
    src_split="train",
    src_samples=1000,
    block_idx=0,
    tgt_samples=50,
    num_calib=100,
    alpha=0.05,
    image_size=512,
    batch_size=16,
):
    """
    Estimate the empirical MMD threshold τ_alpha (false positive rate = alpha)
    by repeatedly sampling target subsets from the *same dataset*.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # Initialize pretrained encoder
    # ----------------------------
    print("Initializing pretrained autoencoder...")
    model = ConvAutoencoderFC(latent_dim=512, pretrained=True).to(device)
    model.eval()

    # ----------------------------
    # Load or extract source features (exact same as feature_extractor.py)
    # ----------------------------
    os.makedirs("features", exist_ok=True)
    src_path = f"features/{dataset_name}_{src_split}_{src_samples}_{block_idx}.npy"

    if os.path.exists(src_path):
        print(f"[INFO] Loaded precomputed source features → {src_path}")
        src_feats = np.load(src_path)
    else:
        print(f"[INFO] Extracting source features ({src_samples}) ...")
        src_loader = get_src_dataloader(
            dataset_name, src_split, batch_size, image_size, src_samples, block_idx
        )
        src_feats = extract_features(model, src_loader, device)
        np.save(src_path, src_feats)
        print(f"[SAVED] {src_path} ({src_feats.shape})")

    print(f"[INFO] Source features shape: {src_feats.shape}")

    # ----------------------------
    # Calibrate null distribution
    # ----------------------------
    null_stats = []

    print(f"\n[INFO] Calibrating null distribution using {num_calib} random runs...")
    for _ in trange(num_calib, desc="Calibration iterations"):
        # Randomly sample target subset (same domain)
        tgt_loader = get_random_dataloader(
            dataset_name, src_split, batch_size, image_size, tgt_samples
        )
        tgt_feats = extract_features(model, tgt_loader, device)

        # Compute MMD statistic
        t_stat, _ = mmd_test(src_feats, tgt_feats)
        null_stats.append(t_stat)

    null_stats = np.array(null_stats)

    # ----------------------------
    # Compute empirical threshold
    # ----------------------------
    tau_alpha = np.percentile(null_stats, 100 * (1 - alpha))
    print(f"\n[RESULT] Empirical threshold τ({1 - alpha:.2f}) = {tau_alpha:.6f}")
    print(f"[RESULT] Mean MMD (no shift): {null_stats.mean():.6f} ± {null_stats.std():.6f}")

    np.save("features/calibration_null_mmd.npy", null_stats)
    print("[SAVED] Null MMD statistics → features/calibration_null_mmd.npy")

    return tau_alpha, null_stats


# --------------------------------------------------------
# Main
# --------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Empirical MMD threshold calibration (same pipeline as feature_extractor).")

    parser.add_argument("--dataset_name", type=str, default="CULane", help="Dataset name (e.g., CULane, Curvelanes)")
    parser.add_argument("--src_split", type=str, default="train", help="Source split (train/test/valid)")
    parser.add_argument("--src_samples", type=int, default=1000, help="Number of source samples")
    parser.add_argument("--block_idx", type=int, default=0, help="Subset index (0 → first 1k)")
    parser.add_argument("--tgt_samples", type=int, default=50, help="Target samples per iteration")
    parser.add_argument("--num_calib", type=int, default=10, help="Number of calibration iterations")
    parser.add_argument("--alpha", type=float, default=0.05, help="False positive rate for threshold")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--image_size", type=int, default=512, help="Resize images to this size")

    args = parser.parse_args()
    tau, null_stats = calibrate_threshold(
        dataset_name=args.dataset_name,
        src_split=args.src_split,
        src_samples=args.src_samples,
        block_idx=args.block_idx,
        tgt_samples=args.tgt_samples,
        num_calib=args.num_calib,
        alpha=args.alpha,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
