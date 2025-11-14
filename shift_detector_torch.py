# shift_detector_torch.py
import os
import argparse
import numpy as np
import torch
from mmd_test import mmd_test  # returns (mmd_statistic, p_value)


class ShiftDetectorTorch:
    """
    PyTorch MMD-based Shift Detector using an empirically calibrated threshold τ.
    Compare raw MMD statistic directly to τ instead of using the bootstrap p-value.
    """

    def __init__(self, tau_threshold=None):
        """
        Args:
            tau_threshold (float): empirical 95th percentile threshold τ from calibration.
        """
        self.tau_threshold = tau_threshold

    def detect_shift(self, src_feats, tgt_feats, sample=None):
        # Optionally subsample for computational control
        if sample is not None:
            src_feats = src_feats[:sample]
            tgt_feats = tgt_feats[:sample]

        # Run MMD test → returns (statistic, p_val) but we ignore p_val
        mmd_val, _ = mmd_test(src_feats, tgt_feats)

        print(f"\n--- MMD Two-Sample Test ---")
        print(f"MMD statistic: {mmd_val:.6f}")
        print(f"Calibrated threshold τ(0.95): {self.tau_threshold:.6f}")

        # Compare to empirical threshold τ
        if mmd_val > self.tau_threshold:
            print("✅ Significant shift detected (MMD > τ)")
            shift_detected = True
        else:
            print("❌ No shift detected (MMD ≤ τ)")
            shift_detected = False

        return shift_detected, mmd_val


def main():
    # ----------------------------
    # Parse arguments
    # ----------------------------
    parser = argparse.ArgumentParser(
        description="MMD-based shift detector with configurable source/target paths."
    )
    parser.add_argument(
        "--src_path",
        type=str,
        default="features/CULane_train_1000_0.npy",
        help="Path to source feature file.",
    )
    parser.add_argument(
        "--tgt_prefix",
        type=str,
        default="features/CULane_train_50_run",
        help="Prefix of target feature files.",
    )
    parser.add_argument(
        "--num_runs", type=int, default=1, help="Number of target runs to evaluate."
    )
    parser.add_argument(
        "--tau_threshold",
        type=float,
        default=0.028562,
        help="Empirical 95th percentile threshold τ.",
    )
    args = parser.parse_args()

    # ----------------------------
    # Load pre-extracted source features
    # ----------------------------
    if not os.path.exists(args.src_path):
        raise FileNotFoundError(f"Source features not found: {args.src_path}")

    print("Loading source features...")
    src_feats = np.load(args.src_path)
    print(f"SRC features: {src_feats.shape}")

    detector = ShiftDetectorTorch(tau_threshold=args.tau_threshold)

    # ----------------------------
    # Evaluate across multiple random targets
    # ----------------------------
    results = []
    for run_idx in range(args.num_runs):
        tgt_path = f"{args.tgt_prefix}{run_idx}.npy"
        if not os.path.exists(tgt_path):
            raise FileNotFoundError(f"Source features not found: {tgt_path}")

        tgt_feats = np.load(tgt_path)
        print(f"\n[RUN {run_idx+1}/{args.num_runs}] TGT: {tgt_feats.shape}")

        shift_detected, mmd_val = detector.detect_shift(src_feats, tgt_feats)
        results.append((shift_detected, mmd_val))

    # ----------------------------
    # Summary statistics
    # ----------------------------
    detections = sum([1 for r, _ in results if r])
    mean_mmd = np.mean([m for _, m in results])
    print(
        f"\n✅ {detections}/{len(results)} shifts detected ({detections/len(results)*100:.2f}%)"
    )
    print(f"📊 Mean MMD: {mean_mmd:.6f}")


if __name__ == "__main__":
    main()
