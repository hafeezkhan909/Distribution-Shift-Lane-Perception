#!/usr/bin/env python3
"""
Visualize Mixed Shift Experiment Results (FLAGGED LOGS ONLY)
- Filters for logs containing: "19792893109391"
- Groups experiments by parsing FILENAMES instead of folders.
- Graphs: 4 Panels (TPR, Raw MMD, Delta MMD, Sample Composition)
- X-Axis Flow: 100% Source (Clean) -> 100% Target (Shifted)
"""

import re
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Global Configuration ---
TARGET_FLAG = "19792893109391"

plt.rcParams["figure.figsize"] = (28, 10)
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)

ARCH_STRINGS = {
    "d128rel": "4096 -> 1024 -> 128",
    "d64rel": "4096 -> 1024 -> 64",
    "d32rel": "4096 -> 1024 -> 32",
    "d128gdd": "4096 -> 512 -> 128",
    "d64gdd": "4096 -> 512 -> 64",
    "d32gdd": "4096 -> 512 -> 32",
    "d64ids": "4096 -> 1024 -> 256 -> 64",
    "d128ids": "4096 -> 1024 -> 256 -> 128",
    "orig": "4096 -> 1024 -> 256 -> 128",
    "d32": "4096 -> 1024 -> 256 -> 32",
}


def get_method_name(arch_key):
    k = arch_key.lower()
    if "rel" in k:
        return "Remove Extra Layer"
    if "gdd" in k:
        return "Gradually Decrease Dimensions"
    if "ids" in k:
        return "Increase Dimensionality"
    if "orig" in k:
        return "Original"
    if "d32" == k:
        return "Base 32"
    return "Unknown Architecture"


# --- Parsing Logic ---
def parse_log_file(log_path):
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # --- CRITICAL FLAG CHECK ---
        if TARGET_FLAG not in content:
            return None

    except Exception:
        return None

    # Parse filename for grouping (e.g., d128rel_K100_Exp1.log)
    filename = log_path.name
    match = re.match(r"^([a-zA-Z0-9]+)_K(\d+)_Exp(\d+)", filename)
    if not match:
        return None

    config_str = match.group(1)
    k_val = int(match.group(2))
    exp_num = int(match.group(3))

    metrics = {
        "config_str": config_str,
        "sample_size": k_val,
        "exp_num": exp_num,
    }

    # Extract Data
    tau = re.search(r"\[RESULT\] τ\([\d.]+\) = ([\d.]+)", content)
    mmd = re.search(r"Average MMD: ([\d.]+) ± ([\d.]+)", content)
    tpr = re.search(r"TPR \(true positive rate\) over \d+ runs: ([\d.]+)%", content)
    shift = re.search(
        r"\[STEP 3\] Data Shift Test:.*?\((\d+)\).*?Curvelanes.*?\((\d+)\).*?CULane",
        content,
    )

    if tau:
        metrics["tau"] = float(tau.group(1))
    if mmd:
        metrics["avg_mmd"], metrics["std_mmd"] = float(mmd.group(1)), float(
            mmd.group(2)
        )
    if tpr:
        metrics["tpr"] = float(tpr.group(1))
    if shift:
        metrics["tgt_samples"], metrics["src_samples"] = int(shift.group(1)), int(
            shift.group(2)
        )

    return metrics if "avg_mmd" in metrics else None


def load_all_data(base_path):
    grouped_data = defaultdict(list)
    valid_logs = 0

    print(f"Scanning for logs with flag: {TARGET_FLAG}...")

    for path in Path(base_path).rglob("*.log"):
        metrics = parse_log_file(path)
        if metrics:
            group_key = (metrics["config_str"], metrics["sample_size"])
            grouped_data[group_key].append(metrics)
            valid_logs += 1

    print(
        f"  > Found {valid_logs} valid logs grouped into {len(grouped_data)} configurations."
    )
    return grouped_data


# --- Drawing Logic ---
def plot_comprehensive_analysis(group_key, experiments, output_dir):
    if not experiments:
        return

    config_str, sample_size = group_key

    # --- CRITICAL FIX: Sort by Source Samples Descending (100% Source -> 100% Target) ---
    experiments.sort(key=lambda x: x.get("src_samples", 0), reverse=True)

    # Re-index the IDs purely for aesthetic graphing (1 to N)
    for i, exp in enumerate(experiments):
        exp["id"] = i + 1
    # -------------------------------------------------------------------------------------

    n = len(experiments)
    x = np.arange(n)

    tprs = [e.get("tpr", 0) for e in experiments]
    mmds = [e.get("avg_mmd", 0) for e in experiments]
    taus = [e.get("tau", 0) for e in experiments]
    delta_mmds = [m - t for m, t in zip(mmds, taus)]
    stds = [e.get("std_mmd", 0) for e in experiments]
    srcs = [e.get("src_samples", 0) for e in experiments]
    tgts = [e.get("tgt_samples", 0) for e in experiments]
    ids = [e.get("id", 0) for e in experiments]

    pretty_arch = get_method_name(config_str)

    # Extract Dims from config string
    dim_match = re.search(r"d(\d+)", config_str)
    dims = (
        int(dim_match.group(1)) if dim_match else (32 if "d32" in config_str else 128)
    )

    arch_str = ARCH_STRINGS.get(config_str, "Architecture Not Found")

    fig, (ax1, ax_raw, ax2, ax3) = plt.subplots(1, 4, figsize=(28, 9))

    # --- Panel 1: TPR ---
    colors = ["#06A77D" if t == 100 else "#D62828" for t in tprs]
    bars1 = ax1.bar(x, tprs, color=colors, alpha=0.7, edgecolor="black", linewidth=1.2)
    ax1.bar_label(bars1, fmt="%.0f", padding=3, fontsize=10, fontweight="bold")
    ax1.set_title("True Positive Rate (%)", fontweight="bold", fontsize=14)
    ax1.set_ylabel("Percentage (%)", fontsize=12)
    ax1.set_ylim(0, 115)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{i}" for i in ids])
    ax1.set_xlabel("Experiment Number", fontsize=11)

    # --- Panel 2: Raw MMD vs Tau ---
    ax_raw.errorbar(
        x, mmds, yerr=stds, fmt="o-", color="#2E86AB", label="Avg MMD", capsize=4
    )
    ax_raw.plot(x, taus, "r--", label=r"Threshold ($\tau$)", linewidth=2)
    ax_raw.set_title("Raw MMD vs. Threshold", fontweight="bold", fontsize=14)
    ax_raw.set_ylabel("MMD Value", fontsize=12)
    ax_raw.legend(loc="upper left", frameon=True)
    ax_raw.set_xticks(x)
    ax_raw.set_xticklabels([f"{i}" for i in ids])
    ax_raw.set_xlabel("Experiment Number", fontsize=11)

    # --- Panel 3: Delta MMD ---
    ax2.errorbar(
        x,
        delta_mmds,
        yerr=stds,
        fmt="o-",
        color="#2E86AB",
        linewidth=2,
        markersize=8,
        label=r"$\Delta$ MMD",
        capsize=4,
    )
    ax2.axhline(
        0, color="black", linestyle="--", linewidth=2, label=r"Threshold ($\tau$)"
    )
    ax2.set_title(r"$\Delta$ MMD (Avg MMD - $\tau$)", fontweight="bold", fontsize=14)
    ax2.set_ylabel(r"$\Delta$ MMD", fontsize=12)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{i}" for i in ids])
    ax2.set_xlabel("Experiment Number", fontsize=11)

    # --- Panel 4: Sample Composition ---
    width = 0.6
    bars_src = ax3.bar(
        x,
        srcs,
        width,
        label="Source (Clean)",
        color="#06A77D",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    bars_tgt = ax3.bar(
        x,
        tgts,
        width,
        bottom=srcs,
        label="Target (Shift)",
        color="#F18F01",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax3.bar_label(
        bars_src,
        label_type="center",
        fmt="%.0f",
        color="white",
        fontweight="bold",
        fontsize=9,
    )
    ax3.bar_label(
        bars_tgt,
        label_type="center",
        fmt="%.0f",
        color="black",
        fontweight="bold",
        fontsize=9,
    )
    ax3.set_title("Sample Composition (Stacked)", fontweight="bold", fontsize=14)
    ax3.set_ylabel("Samples", fontsize=12)

    ax3.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{i}" for i in ids])
    ax3.set_xlabel("Experiment Number", fontsize=11)

    # --- Layout & Titles ---
    plt.suptitle("Distribution Shift Study", fontsize=26, fontweight="bold", y=0.98)
    subtitle_str = (
        f"{pretty_arch} ({config_str}) | {dims} Dimensions | Sample Size: {sample_size}"
    )
    plt.figtext(
        0.5,
        0.90,
        subtitle_str,
        ha="center",
        fontsize=18,
        fontstyle="italic",
        color="#333333",
    )

    # Architecture Footer
    fig.text(
        0.5,
        0.05,
        f"Network Architecture: {arch_str}",
        ha="center",
        fontsize=14,
        fontweight="bold",
        bbox=dict(
            facecolor="#FBFCFC", alpha=0.9, edgecolor="#AEB6BF", boxstyle="round,pad=1"
        ),
    )

    plt.tight_layout(rect=[0.02, 0.10, 0.98, 0.91])

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save with a clear filename
    save_name = f"{config_str}_K{sample_size}_summary.png"
    plt.savefig(out_path / save_name, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path", type=str, required=True, help="Root folder to scan"
    )
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

    grouped_data = load_all_data(args.base_path)

    if not grouped_data:
        print(f"No log files found containing the flag: {TARGET_FLAG}")
        return

    for group_key, experiments in grouped_data.items():
        if len(experiments) > 0:
            plot_comprehensive_analysis(group_key, experiments, args.output_dir)
            config_str, sample_size = group_key
            print(f"✓ Generated Graph: {config_str}_K{sample_size}")


if __name__ == "__main__":
    main()
