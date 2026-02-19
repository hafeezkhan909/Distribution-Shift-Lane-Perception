#!/usr/bin/env python3
"""
Generate Summary Table for FLAGGED Logs Only
- Scans directory tree for ALL .log files.
- Filters for logs containing the specific flag: "19792893109391"
- Groups experiments by parsing their FILENAMES.
- Sorted by: Stable 100% -> Lowest Samples -> Lowest Dims -> Lowest Std
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
    if "orig" in k: return "Original"
    if "d32" == k: return "Base 32"
    return "Unknown Architecture"

# --- Parsing Logic ---
def parse_log_file(log_path):
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # --- CRITICAL FLAG CHECK ---
        if TARGET_FLAG not in content:
            return None
        # ---------------------------

        # Parse filename to get grouping info
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

        # Extract Metrics
        tau = re.search(r'\[RESULT\] τ\([\d.]+\) = ([\d.]+)', content)
        tpr = re.search(r'TPR \(true positive rate\) over \d+ runs: ([\d.]+)%', content)
        shift = re.search(r'\[STEP 3\] Data Shift Test:.*?\((\d+)\).*?Curvelanes.*?\((\d+)\).*?CULane', content)
        
        if tau: metrics['tau'] = float(tau.group(1))
        if tpr: metrics['tpr'] = float(tpr.group(1))
        if shift: 
            metrics['tgt_samples'] = int(shift.group(1))
            metrics['src_samples'] = int(shift.group(2))
            
        return metrics if 'tpr' in metrics else None

    except Exception:
        return None

def load_all_data(base_path):
    grouped_data = defaultdict(list)
    valid_logs = 0

    print(f"Scanning for logs with flag: {TARGET_FLAG}...")
    
    for path in Path(base_path).rglob('*.log'):
        metrics = parse_log_file(path)
        if metrics:
            group_key = (metrics['config_str'], metrics['sample_size'])
            grouped_data[group_key].append(metrics)
            valid_logs += 1

    print(f"  > Found {valid_logs} valid logs grouped into {len(grouped_data)} configurations.")
    return grouped_data

# --- Analysis Logic ---
def analyze_group(group_key, experiments):
    config_str, k_val = group_key
    
    # Sort by Source Samples Descending
    experiments.sort(key=lambda x: x.get('src_samples', 0), reverse=True)
    
    tprs = [e.get('tpr', 0) for e in experiments]
    taus = [e.get('tau', 0) for e in experiments]
    
    first_100 = next((i + 1 for i, t in enumerate(tprs) if t >= 100), None)
    
    const_100 = None
    for i in range(len(tprs)):
        if all(t >= 100 for t in tprs[i:]):
            const_100 = i + 1
            break
            
    # Calculate Mean Tau and Std
    tau_mean = np.mean(taus) if taus else 0.0
    tau_std = np.std(taus) if taus else 0.0
    
    dim_match = re.search(r'd(\d+)', config_str)
    dims = int(dim_match.group(1)) if dim_match else (32 if 'd32' in config_str else 128)

    return {
        "Method": get_method_name(config_str),
        "Dims": dims,
        "Samples": k_val,
        "First 100%": first_100,
        "Stable 100%": const_100,
        "Tau Std": tau_std,
        "Valid Logs": len(experiments)
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

    # Sort by: Stable 100% -> Samples -> Dims -> Tau Std
    data.sort(key=lambda x: (
        x['Stable 100%'] if x['Stable 100%'] is not None else 999,
        x['Samples'],
        x['Dims'],
        x['Tau Std']
    ))

    cols = ["Method", "Dims", "Samples", "First 100%\n(Speed)", "Stable 100%\n(Reliability)", "Tau Std\n(Stability)"]
    
    first_vals = [x['First 100%'] for x in data if x['First 100%'] is not None]
    stable_vals = [x['Stable 100%'] for x in data if x['Stable 100%'] is not None]
    tau_std_vals = [x['Tau Std'] for x in data]
    
    min_first, max_first = (min(first_vals), max(first_vals)) if first_vals else (0,1)
    min_stable, max_stable = (min(stable_vals), max(stable_vals)) if stable_vals else (0,1)
    min_tau_std, max_tau_std = (min(tau_std_vals), max(tau_std_vals)) if tau_std_vals else (0,1)

    rows = len(data)
    fig_height = max(4, rows * 0.6 + 2) 
    fig, ax = plt.subplots(figsize=(15, fig_height))
    ax.axis('off')
    
    cell_text = []
    cell_colours = []
    row_colors = ['#FFFFFF', '#F4F6F7']

    for idx, row in enumerate(data):
        row_txt = []
        row_col = []
        bg_base = row_colors[idx % 2]
        
        row_txt.append(row['Method'])
        row_col.append(bg_base)
        
        row_txt.append(str(row['Dims']))
        row_col.append(bg_base)
        
        s_size = row['Samples']
        s_str = f"{s_size}" if s_size > 0 else "?"
        row_txt.append(s_str)
        row_col.append(bg_base)
        
        # First 100%
        val = row['First 100%']
        txt = f"Exp {val}" if val else "Never"
        col = get_pastel_color(val, min_first, max_first) if val else "#E5E8E8"
        row_txt.append(txt)
        row_col.append(col)
        
        # Stable 100%
        val = row['Stable 100%']
        txt = f"Exp {val}" if val else "Never"
        col = get_pastel_color(val, min_stable, max_stable) if val else "#E5E8E8"
        row_txt.append(txt)
        row_col.append(col)
        
        # Tau Std
        val = row['Tau Std']
        txt = f"{val:.5f}"
        col = get_pastel_color(val, min_tau_std, max_tau_std) 
        row_txt.append(txt)
        row_col.append(col)
        
        cell_text.append(row_txt)
        cell_colours.append(row_col)

    table = ax.table(cellText=cell_text, colLabels=cols, cellColours=cell_colours, 
                     loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
    
    plt.rcParams['font.family'] = 'sans-serif'

    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('#BDC3C7')
        cell.set_linewidth(0.5)
        
        if i == 0:
            cell.set_text_props(weight='bold', color='#FFFFFF', fontsize=11)
            cell.set_facecolor('#2C3E50')
            cell.set_height(0.08)
        else:
            cell.set_text_props(fontsize=10, color='#212F3C')
            cell.set_height(0.06)
            
            record = data[i-1]
            is_best = False
            if j == 3 and record['First 100%'] == min_first: is_best = True
            if j == 4 and record['Stable 100%'] == min_stable: is_best = True
            
            if is_best:
                cell.set_text_props(weight='bold', fontsize=10, color='#000000')
                cell.set_linewidth(1.5)
                cell.set_edgecolor('#1E8449')

    plt.suptitle("Architecture Robustness Summary", 
                 fontweight='bold', fontsize=16, y=1.02, color='#2C3E50')
    plt.title(f"Sorted by: Stable 100%, Samples, Dims, Std", 
              fontsize=10, color='#E74C3C', y=1.0)
    
    plt.tight_layout()
    save_loc = Path(output_path) / "flagged_summary_table.png"
    plt.savefig(save_loc, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"✓ Flagged table generated at: {save_loc}")

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', type=str, required=True, help="Root folder to scan")
    parser.add_argument('--output-dir', type=str, default='figures')
    args = parser.parse_args()

    grouped_data = load_all_data(args.base_path)
    
    all_data = []
    for group_key, experiments in grouped_data.items():
        if len(experiments) > 0:
            stats = analyze_group(group_key, experiments)
            all_data.append(stats)

    if all_data:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        draw_pretty_table(all_data, args.output_dir)
    else:
        print(f"No log files found containing the flag: {TARGET_FLAG}")

if __name__ == "__main__":
    main()
