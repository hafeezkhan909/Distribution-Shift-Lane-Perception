#!/usr/bin/env python3
"""
Generate LaTeX-Style Summary Table for FLAGGED Logs
- Scans directory tree for FLAGGED logs.
- Renders a publication-ready booktabs-style PNG table using a pure coordinate grid.
- Guaranteed zero text overlap.
"""

import re
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# Use Serif font to mimic LaTeX's default Computer Modern / Times appearance
plt.rcParams['font.family'] = 'serif'

# --- Configuration ---
TARGET_FLAG = "19792893109391"

# --- Naming Helper ---
def get_method_name(arch_key):
    k = arch_key.lower()
    if "rel" in k: return "Remove Extra Layer"
    if "gdd" in k: return "Gradually Decrease Dimensions"
    if "ids" in k: return "Increase Dimensionality"
    if "orig" in k or "d32" == k: return "Original"
    return "Unknown Architecture"

# --- Parsing Logic ---
def parse_log_file(log_path):
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        if TARGET_FLAG not in content:
            return None

        filename = log_path.name
        match = re.match(r'^([a-zA-Z0-9]+)_K(\d+)_Exp(\d+)', filename)
        if not match:
            return None
        
        config_str = match.group(1)
        k_val = int(match.group(2))
        exp_num = int(match.group(3))

        metrics = {
            'config_str': config_str,
            'sample_size': k_val,
            'exp_num': exp_num,
        }

        tpr = re.search(r'TPR \(true positive rate\) over \d+ runs: ([\d.]+)%', content)
        shift = re.search(r'\[STEP 3\] Data Shift Test:.*?\((\d+)\).*?Curvelanes.*?\((\d+)\).*?CULane', content)
        
        if tpr: metrics['tpr'] = float(tpr.group(1))
        if shift: 
            metrics['src_samples'] = int(shift.group(2))
            
        return metrics if 'tpr' in metrics else None
    except Exception:
        return None

def load_all_data(base_path):
    grouped_data = defaultdict(list)
    for path in Path(base_path).rglob('*.log'):
        metrics = parse_log_file(path)
        if metrics:
            group_key = (metrics['config_str'], metrics['sample_size'])
            grouped_data[group_key].append(metrics)
    return grouped_data

# --- Analysis Logic ---
def analyze_group(group_key, experiments):
    config_str, k_val = group_key
    
    experiments.sort(key=lambda x: x.get('src_samples', 0), reverse=True)
    tprs = [e.get('tpr', 0) for e in experiments]
    
    def get_stable_idx(threshold):
        for i in range(len(tprs)):
            if all(t >= threshold for t in tprs[i:]):
                return i + 1
        return None

    dim_match = re.search(r'd(\d+)', config_str)
    dims = int(dim_match.group(1)) if dim_match else (32 if 'd32' in config_str else 128)

    return {
        "Method": get_method_name(config_str),
        "Dims": dims,
        "Samples": k_val,
        "Stable 100%": get_stable_idx(100),
        "Stable 75%": get_stable_idx(75),
        "Stable 50%": get_stable_idx(50),
        "Stable 25%": get_stable_idx(25)
    }

# --- Pure Coordinate Drawing Logic ---
def draw_latex_style_table(all_data, output_dir):
    methods_dict = defaultdict(list)
    for entry in all_data:
        methods_dict[entry['Method']].append(entry)

    methods_order = ["Gradually Decrease Dimensions"]
    methods = [m for m in methods_order if m in methods_dict]

    # Calculate exact vertical units needed for the figure
    ROW_H = 1.0
    total_y_units = 2.0  # Toprule + Header + Midrule spacing
    for m in methods:
        total_y_units += 1.8  # Method Header + Midrule spacing
        total_y_units += len(methods_dict[m]) * ROW_H # Data rows
    total_y_units += 0.5 # Bottom rule padding

    # Setup the physical figure bounds to match the units
    fig_height = (total_y_units * 0.25) + 0.5 
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_y_units)

    fig.suptitle("Architecture Robustness Summary by Technique", fontweight='bold', fontsize=14, y=0.98)

    cols = ["Latent Space\nDims", "Samples", "Stable 100%", "Stable 75%", "Stable 50%", "Stable 25%"]
    col_x = [0.08, 0.22, 0.40, 0.58, 0.76, 0.94] # Rebalanced for 6 columns

    current_y = total_y_units - 0.5

    # 1. Toprule
    ax.plot([0, 1], [current_y, current_y], color='black', lw=1.5, clip_on=False)
    current_y -= ROW_H * 0.8

    # 2. Main Headers
    for x, header in zip(col_x, cols):
        ax.text(x, current_y, header, ha='center', va='center', weight='bold', fontsize=11, clip_on=False)
    current_y -= ROW_H * 0.8

    # 3. Midrule
    ax.plot([0, 1], [current_y, current_y], color='black', lw=1.0, clip_on=False)

    for method in methods:
        data = methods_dict[method]
        
        # Sort Best -> Worst based on new stability thresholds
        data.sort(key=lambda x: (
            x['Stable 100%'] if x['Stable 100%'] is not None else 999,
            x['Stable 75%'] if x['Stable 75%'] is not None else 999,
            x['Stable 50%'] if x['Stable 50%'] is not None else 999,
            x['Stable 25%'] if x['Stable 25%'] is not None else 999,
            -x['Samples'], 
            -x['Dims']
        ))

        def get_min(key):
            vals = [x[key] for x in data if x[key] is not None]
            return min(vals) if vals else None

        min_100 = get_min('Stable 100%')
        min_75 = get_min('Stable 75%')
        min_50 = get_min('Stable 50%')
        min_25 = get_min('Stable 25%')

        # 4. Method Header
        current_y -= ROW_H * 1.0
        ax.text(0.5, current_y, method, ha='center', va='center', weight='bold', style='italic', fontsize=11, clip_on=False)
        current_y -= ROW_H * 0.6

        # 5. Method Midrule
        ax.plot([0, 1], [current_y, current_y], color='black', lw=0.75, clip_on=False)

        # 6. Data Rows
        for row in data:
            current_y -= ROW_H
            
            row_txt = [
                str(row['Dims']),
                str(row['Samples']),
                f"Exp {row['Stable 100%']}" if row['Stable 100%'] else "Never",
                f"Exp {row['Stable 75%']}" if row['Stable 75%'] else "Never",
                f"Exp {row['Stable 50%']}" if row['Stable 50%'] else "Never",
                f"Exp {row['Stable 25%']}" if row['Stable 25%'] else "Never"
            ]
            
            weights = [
                'normal', 
                'normal', 
                'bold' if (row['Stable 100%'] == min_100 and min_100 is not None) else 'normal',
                'bold' if (row['Stable 75%'] == min_75 and min_75 is not None) else 'normal',
                'bold' if (row['Stable 50%'] == min_50 and min_50 is not None) else 'normal',
                'bold' if (row['Stable 25%'] == min_25 and min_25 is not None) else 'normal'
            ]

            for x, txt, weight in zip(col_x, row_txt, weights):
                ax.text(x, current_y, txt, ha='center', va='center', weight=weight, fontsize=10, clip_on=False)

    # 7. Bottom rule
    current_y -= ROW_H * 0.7
    ax.plot([0, 1], [current_y, current_y], color='black', lw=1.5, clip_on=False)

    plt.tight_layout()
    save_loc = Path(output_dir) / "latex_summary_table_gdd.png"
    fig.savefig(save_loc, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"✓ Generated absolute-grid LaTeX table at: {save_loc}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='figures')
    args = parser.parse_args()

    grouped_data = load_all_data(args.base_path)
    all_data = [analyze_group(k, v) for k, v in grouped_data.items() if len(v) > 0]

    if all_data:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        draw_latex_style_table(all_data, args.output_dir)
    else:
        print("No log files found containing the target flag.")

if __name__ == "__main__":
    main()
