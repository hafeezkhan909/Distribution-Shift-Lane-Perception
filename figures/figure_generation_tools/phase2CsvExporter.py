#!/usr/bin/env python3
"""
Phase 2 CSV Exporter
--------------------
Reads all Phase 2 JSON experiment result files and writes a summary CSV with
three side-by-side sections: TPR (%), Tau Threshold, and Avg MMD.

Layout matches the requested spreadsheet format:
  - Rows:    8 distribution configs (4 Train + 4 Test) × 3 sample sizes (1000, 100, 10)
  - Columns: 4 autoencoder models (ImageNet, Random, CULane, CurveLanes)
  - Three sections (TPR | Tau Threshold | Avg MMD) placed side-by-side in 18 columns

"Train" rows are read from --train-log-dir (default: logs/ModelExperiments/Phase2/).
"Test"  rows are read from --test-log-dir  (default: logs/Phase2_test/).
Missing files leave the corresponding cells empty.

Usage:
  python phase2CsvExporter.py \\
      --train-log-dir  ../logs/ModelExperiments/Phase2 \\
      --test-log-dir   ../logs/Phase2_test \\
      --output         ../ModelExperimentFigures/Phase2/phase2_summary.csv
"""

import os
import json
import argparse
import csv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model strings as used in JSON filenames
MODELS = ["ImageNet", "Random", "CU_Lane", "CurveLanes"]
# Friendly model labels shown in CSV column headers
MODEL_LABELS = ["ImageNet", "Random", "CULane", "Curvelanes"]

SAMPLE_SIZES = [1000, 100, 10]

# Row definitions: (display_label, is_test_row, filename_combo)
# filename_combo is the segment that appears between "{model}Model_" and "Data.json"
# in the JSON filename (e.g. "CULanes" → "P2{K}Samples_{model}Model_CULanesData.json").
ROWS = [
    ("CULane to CULane", False, "CULanes"),
    ("CULane to CULane Test", True, "CULanesTest"),
    ("CULane to Curvelanes", False, "CULanes2Curvelanes"),
    ("CULane to Curvelanes Test", True, "CULanes2CurvelanesTest"),
    ("Curvelanes to CULane", False, "Curvelanes2CULanes"),
    ("Curvelanes to CULane Test", True, "Curvelanes2CULanesTest"),
    ("Curvelanes to Curvelanes", False, "Curvelanes"),
    ("Curvelanes to Curvelanes Test", True, "CurvelanesTest"),
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_experiment(log_dir: str, k: int, model: str, combo: str):
    """
    Load a single JSON result file and return a dict with
    {'tpr', 'tau', 'mmd', 'mmd_std'}, or None if the file is missing/invalid.

    Filename pattern: P2{k}Samples_{model}Model_{combo}Data.json
    """
    if not log_dir or not os.path.isdir(log_dir):
        return None

    fname = f"P2{k}Samples_{model}Model_{combo}Data.json"
    path = os.path.join(log_dir, fname)

    if not os.path.exists(path):
        return None

    try:
        with open(path) as f:
            data = json.load(f)
        d = data["experiments"][0]["data"]
        return {
            "tpr": float(d["Data Shift Test Data"]["TPR"]),
            "tau": float(d["Calibration"]["Result"]["Tau"]),
            "mmd": float(d["Data Shift Test Data"]["Average MMD"]),
            "mmd_std": float(d["Data Shift Test Data"]["Average MMD (std)"]),
        }
    except (KeyError, IndexError, ValueError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def fmt_tpr(val):
    """Format TPR: integer representation when the value is whole, else 1 d.p."""
    if val is None:
        return ""
    return f"{val:g}"


def fmt_tau(val):
    """Format Tau Threshold to 6 decimal places."""
    if val is None:
        return ""
    return f"{val:.6f}"


def fmt_mmd(mmd, mmd_std):
    """Format Average MMD as 'mean ± std' to 6 decimal places."""
    if mmd is None or mmd_std is None:
        return ""
    return f"{mmd:.6f} \u00b1 {mmd_std:.6f}"


# ---------------------------------------------------------------------------
# CSV construction
# ---------------------------------------------------------------------------


def build_csv_rows(train_log_dir: str, test_log_dir: str) -> list:
    """
    Build all CSV rows as lists of strings.

    The CSV has 18 columns, laid out as three 6-column sections:
      Section 1 (TPR):  cols  0-5   [sec_label, row_name, Im, Ra, CU, Cv]
      Section 2 (Tau):  cols  6-11  [sec_label, row_name, Im, Ra, CU, Cv]
      Section 3 (MMD):  cols 12-17  [sec_label, row_name, Im, Ra, CU, Cv]
    """
    all_rows = []

    # ---- Overall section header (one row at the very top) ----
    all_rows.append(
        [
            "TPR",
            "",
            "",
            "",
            "",
            "",
            "Tau Threshold",
            "",
            "",
            "",
            "",
            "",
            "Avg MMD",
            "",
            "",
            "",
            "",
            "",
        ]
    )

    for k_idx, k in enumerate(SAMPLE_SIZES):
        # Blank separator row between K blocks
        if k_idx > 0:
            all_rows.append([""] * 18)

        # K sample-size header
        ks = f"{k} Samples"
        all_rows.append(
            [ks, "", "", "", "", "", ks, "", "", "", "", "", ks, "", "", "", "", ""]
        )

        # "Autoencoders" sub-header (spans the 4 model columns in each section)
        all_rows.append(
            [
                "",
                "",
                "Autoencoders",
                "",
                "",
                "",
                "",
                "",
                "Autoencoders",
                "",
                "",
                "",
                "",
                "",
                "Autoencoders",
                "",
                "",
                "",
            ]
        )

        # Model column header
        all_rows.append(
            ["", ""] + MODEL_LABELS + ["", ""] + MODEL_LABELS + ["", ""] + MODEL_LABELS
        )

        # ---- Data rows ----
        for r_idx, (label, is_test_row, combo) in enumerate(ROWS):
            # "Distributions" label only on the very first data row of the block
            sec_label = "Distributions" if r_idx == 0 else ""

            log_dir = test_log_dir if is_test_row else train_log_dir

            tpr_vals, tau_vals, mmd_vals = [], [], []

            for model in MODELS:
                result = load_experiment(log_dir, k, model, combo)
                if result:
                    tpr_vals.append(fmt_tpr(result["tpr"]))
                    tau_vals.append(fmt_tau(result["tau"]))
                    mmd_vals.append(fmt_mmd(result["mmd"], result["mmd_std"]))
                else:
                    tpr_vals.append("")
                    tau_vals.append("")
                    mmd_vals.append("")

            all_rows.append(
                [sec_label, label]
                + tpr_vals
                + [sec_label, label]
                + tau_vals
                + [sec_label, label]
                + mmd_vals
            )

    return all_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Export Phase 2 experiment results (TPR, Tau, Avg MMD) to a CSV."
    )
    parser.add_argument(
        "--train-log-dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "logs", "ModelExperiments", "Phase2"
        ),
        help="Directory containing Phase 2 Train JSON result files.",
    )
    parser.add_argument(
        "--test-log-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "logs", "Phase2_test"),
        help=(
            "Directory containing Phase 2 Test JSON result files "
            "(produced by LocalBash/run_phase2_test.sh). "
            "Missing files leave those cells empty."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "ModelExperimentFigures",
            "Phase2",
            "phase2_summary.csv",
        ),
        help="Path for the output CSV file.",
    )
    args = parser.parse_args()

    train_log_dir = os.path.abspath(args.train_log_dir)
    test_log_dir = os.path.abspath(args.test_log_dir)
    output_path = os.path.abspath(args.output)

    if not os.path.isdir(train_log_dir):
        print(f"ERROR: --train-log-dir not found: {train_log_dir}")
        return

    if not os.path.isdir(test_log_dir):
        print(
            f"NOTE: --test-log-dir not found: {test_log_dir} "
            "— Test rows will be empty until experiments are run."
        )

    print(f"Reading Train logs from: {train_log_dir}")
    print(f"Reading Test  logs from: {test_log_dir}")

    rows = build_csv_rows(train_log_dir, test_log_dir)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\nCSV written to: {output_path}")


if __name__ == "__main__":
    main()
