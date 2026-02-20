#!/usr/bin/env python3
"""
Generate Summary Table for FLAGGED Logs Only
- Scans directory tree for ALL .log files.
- Filters for logs containing the specific flag: "19792893109391"
- Groups experiments by parsing their FILENAMES.
- Sorted by: Stable 100% -> Lowest Samples -> Lowest Dims -> Lowest Std
- Columns fit content naturally.
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

# --- Configuration ---
TARGET_FLAG = "19792893109391"

# --- Naming Helper ---
def get_method_name(arch_key):
    k = arch_key.lower()
    if "rel" in k: return "Remove Extra Layer"
    if "gdd" in k: return "Gradually Decrease Dimensions"
    if "ids" in k: return "Increase Dimensionality"
    if "orig" in k or "d32" == k: return "Original" # Updated mapping
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

        tau = re.search(r'\[RESULT\] τ\([\d.]+\) = ([\d.]+)', content)
        tpr = re.search(r'TPR \(true positive rate\) over \d+ runs: ([\d.]+)%', content)
        shift = re.search(r'\[STEP 3\] Data Shift Test:.*?\((\d+)\).*?Curvelanes.*?\((\d+)\).*?CULane', content)
        
        if tau: metrics['tau'] = float(tau.group(1))
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
    taus = [e.get('tau', 0) for e in experiments]
    
    first_100 = next((i + 1 for i, t in enumerate(tprs) if t >= 100), None)
    const_100 = None
    for i in range(len(tprs)):
        if all(t >= 100 for t in tprs[i:]):
            const_100 = i + 1
            break
            
    dim_match = re.search(r'd(\d+)', config_str)
    dims = int(dim_match.group(1)) if dim_match else (32 if 'd32' in config_str else 128)

    return {
        "Method": get_method_name(config_str),
        "Dims": dims,
        "Samples": k_val,
        "First 100%": first_100,
        "Stable 100%": const_100,
        "Tau Std": np.std(taus) if taus else 0.0
    }

# --- Drawing Logic ---
def get_pastel_color(value, min_val, max_val, invert=False):
    if value is None: return "#F2F3F4" 
    if min_val == max_val: return "#ffffff"
    norm = (value - min_val) / (max_val - min_val)
    if invert: norm = 1 - norm 
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#abebc6", "#f9e79f", "#f5b7b1"])
    return matplotlib.colors.to_hex(cmap(norm))

def draw_pretty_table(data, output_path):
    if not data: return

    data.sort(key=lambda x: (
        x['Stable 100%'] if x['Stable 100%'] is not None else 999,
        x['Samples'], x['Dims'], x['Tau Std']
    ))

    cols = ["Method", "Dims", "Samples", "First 100%\n(Speed)", "Stable 100%\n(Reliability)", "Tau Std\n(Stability)"]
    
    first_vals = [x['First 100%'] for x in data if x['First 100%'] is not None]
    stable_vals = [x['Stable 100%'] for x in data if x['Stable 100%'] is not None]
    tau_std_vals = [x['Tau Std'] for x in data]
    
    min_first, max_first = (min(first_vals), max(first_vals)) if first_vals else (0,1)
    min_stable, max_stable = (min(stable_vals), max(stable_vals)) if stable_vals else (0,1)
    min_tau_std, max_tau_std = (min(tau_std_vals), max(tau_std_vals)) if tau_std_vals else (0,1)

    rows = len(data)
    fig_height = max(4, rows * 0.5 + 2) 
    fig, ax = plt.subplots(figsize=(15, fig_height))
    ax.axis('off')
    
    cell_text = []
    cell_colours = []
    row_colors = ['#FFFFFF', '#F4F6F7']

    for idx, row in enumerate(data):
        bg_base = row_colors[idx % 2]
        
        row_txt = [
            row['Method'],
            str(row['Dims']),
            str(row['Samples']),
            f"Exp {row['First 100%']}" if row['First 100%'] else "Never",
            f"Exp {row['Stable 100%']}" if row['Stable 100%'] else "Never",
            f"{row['Tau Std']:.5f}"
        ]
        
        row_col = [
            bg_base, bg_base, bg_base,
            get_pastel_color(row['First 100%'], min_first, max_first) if row['First 100%'] else "#E5E8E8",
            get_pastel_color(row['Stable 100%'], min_stable, max_stable) if row['Stable 100%'] else "#E5E8E8",
            get_pastel_color(row['Tau Std'], min_tau_std, max_tau_std)
        ]
        cell_text.append(row_txt)
        cell_colours.append(row_col)

    table = ax.table(cellText=cell_text, colLabels=cols, cellColours=cell_colours, 
                     loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('#BDC3C7')
        cell.set_linewidth(0.5)
        if i == 0:
            cell.set_text_props(weight='bold', color='#FFFFFF')
            cell.set_facecolor('#2C3E50')
        else:
            record = data[i-1]
            if (j == 3 and record['First 100%'] == min_first) or \
               (j == 4 and record['Stable 100%'] == min_stable):
                cell.set_text_props(weight='bold')
                cell.set_linewidth(1.2)
                cell.set_edgecolor('#1E8449')

    plt.suptitle("Architecture Robustness Summary", fontweight='bold', fontsize=16, y=1.02)
    plt.title("Sorted by: Stable 100%, Samples, Dims, Std", fontsize=10, color='#7F8C8D', y=1.0)
    
    # Auto-fit all columns to content
    plt.gcf().canvas.draw()
    table.auto_set_column_width(col=list(range(len(cols))))

    plt.tight_layout()
    save_loc = Path(output_path) / "flagged_summary_table.png"
    plt.savefig(save_loc, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"✓ Table generated at: {save_loc}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='figures')
    args = parser.parse_args()

    grouped_data = load_all_data(args.base_path)
    all_data = [analyze_group(k, v) for k, v in grouped_data.items() if len(v) > 0]

    if all_data:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        draw_pretty_table(all_data, args.output_dir)

if __name__ == "__main__":
    main()
