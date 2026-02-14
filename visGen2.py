#!/usr/bin/env python3
"""
Visualize Mixed Shift Experiment Results
- Title: Distribution Shift Study
- Subtitle: Separated from title, closer to graphs.
- Graphs: Data labels added.
- Legend: Moved below the right graph.
"""

import re
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Global Configuration
plt.rcParams['figure.figsize'] = (22, 10) # Slightly taller to accommodate bottom legend
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

ARCH_STRINGS = {
    "d128rel": "4096 -> 1024 -> 128", "d64rel": "4096 -> 1024 -> 64", "d32rel": "4096 -> 1024 -> 32",
    "d128gdd": "4096 -> 512 -> 128", "d64gdd": "4096 -> 512 -> 64", "d32gdd": "4096 -> 512 -> 32",
    "d64ids": "4096 -> 1024 -> 256 -> 64", "d128ids": "4096 -> 1024 -> 256 -> 128",
    "orig": "4096 -> 1024 -> 256 -> 128", "d32": "4096 -> 1024 -> 256 -> 32",
}

def camel_to_title(text):
    """Converts CamelCase or underscore_strings to Title Case."""
    text = text.replace('_', ' ')
    return re.sub(r'(?<!^)(?<!\s)(?=[A-Z])', ' ', text).strip()

def parse_bash_script(bash_path):
    if not bash_path.exists(): return {}
    try:
        with open(bash_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception: return {}
    
    config = {}
    dconfig_match = re.search(r'--dConfig\s+"([^"]+)"', content)
    dconf = dconfig_match.group(1) if dconfig_match else "orig"
    config['arch_key'] = dconf
    
    dim_match = re.search(r'd(\d+)', dconf)
    if dim_match:
        config['dims'] = dim_match.group(1)
    else:
        config['dims'] = "32" if "d32" in dconf else "128"
    
    return config

def parse_log_file(log_path):
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception: return None
    
    metrics = {}
    k_match = re.search(r'K(\d+)', log_path.stem)
    metrics['sample_size'] = k_match.group(1) if k_match else "Unknown"

    tau = re.search(r'\[RESULT\] τ\([\d.]+\) = ([\d.]+)', content)
    mmd = re.search(r'Average MMD: ([\d.]+) ± ([\d.]+)', content)
    tpr = re.search(r'TPR \(true positive rate\) over \d+ runs: ([\d.]+)%', content)
    shift = re.search(r'\[STEP 3\] Data Shift Test:.*?\((\d+)\).*?Curvelanes.*?\((\d+)\).*?CULane', content)
    
    if tau: metrics['tau'] = float(tau.group(1))
    if mmd: metrics['avg_mmd'], metrics['std_mmd'] = float(mmd.group(1)), float(mmd.group(2))
    if tpr: metrics['tpr'] = float(tpr.group(1))
    if shift: 
        metrics['tgt_samples'], metrics['src_samples'] = int(shift.group(1)), int(shift.group(2))
    
    return metrics if 'avg_mmd' in metrics else None

def load_folder_data(logs_dir):
    experiments = []
    path = Path(logs_dir)
    log_files = sorted([f for f in path.iterdir() if f.is_file() and f.suffix == '.log'])
    
    parent_folder_name = path.parent.name
    
    for log_file in log_files:
        metrics = parse_log_file(log_file)
        if metrics:
            bash_file = path / f"{log_file.stem}.sh"
            metrics['bash'] = parse_bash_script(bash_file)
            metrics['parent_folder'] = parent_folder_name
            experiments.append(metrics)
    
    experiments.sort(key=lambda x: x.get('src_samples', 0), reverse=True)
    
    for i, exp in enumerate(experiments):
        exp['id'] = i + 1
        
    return experiments

def plot_comprehensive_analysis(experiments, output_dir, folder_name):
    if not experiments: return
    n = len(experiments)
    x = np.arange(n)
    
    tprs = [e.get('tpr', 0) for e in experiments]
    mmds = [e.get('avg_mmd', 0) for e in experiments]
    taus = [e.get('tau', 0) for e in experiments]
    delta_mmds = [m - t for m, t in zip(mmds, taus)]
    stds = [e.get('std_mmd', 0) for e in experiments]
    srcs = [e.get('src_samples', 0) for e in experiments]
    tgts = [e.get('tgt_samples', 0) for e in experiments]
    ids = [e.get('id', 0) for e in experiments]
    
    config = experiments[0].get('bash', {})
    raw_title = experiments[0].get('parent_folder', "Unknown")
    pretty_arch = camel_to_title(raw_title)
    dims = config.get('dims', "???")
    sample_size = experiments[0].get('sample_size', "Unknown")
    arch_str = ARCH_STRINGS.get(config.get('arch_key', 'orig'), "Architecture Not Found")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 9))

    # --- Panel 1: TPR ---
    colors = ['#06A77D' if t == 100 else '#D62828' for t in tprs]
    bars1 = ax1.bar(x, tprs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Add labels to TPR bars
    ax1.bar_label(bars1, fmt='%.0f', padding=3, fontsize=10, fontweight='bold')
    
    ax1.set_title('True Positive Rate (%)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_ylim(0, 115)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{i}" for i in ids])
    ax1.set_xlabel("Experiment Number", fontsize=11)

    # --- Panel 2: Delta MMD ---
    ax2.errorbar(x, delta_mmds, yerr=stds, fmt='o-', color='#2E86AB', linewidth=2, markersize=8, label=r'$\Delta$ MMD', capsize=4)
    ax2.axhline(0, color='black', linestyle='--', linewidth=2, label=r'Threshold ($\tau$)')
    
    ax2.set_title(r'$\Delta$ MMD (Avg MMD - $\tau$)', fontweight='bold', fontsize=14)
    ax2.set_ylabel(r'$\Delta$ MMD', fontsize=12)
    ax2.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{i}" for i in ids])
    ax2.set_xlabel("Experiment Number", fontsize=11)

    # --- Panel 3: Sample Composition ---
    width = 0.6
    bars_src = ax3.bar(x, srcs, width, label='Source (Clean)', color='#06A77D', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars_tgt = ax3.bar(x, tgts, width, bottom=srcs, label='Target (Shift)', color='#F18F01', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add labels to Stacked Bars (Centered in the segments)
    # Only label if segment height > 5 to avoid clutter on tiny segments
    ax3.bar_label(bars_src, label_type='center', fmt='%.0f', color='white', fontweight='bold', fontsize=9)
    ax3.bar_label(bars_tgt, label_type='center', fmt='%.0f', color='black', fontweight='bold', fontsize=9)

    ax3.set_title('Sample Composition (Stacked)', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Samples', fontsize=12)
    
    # Legend BELOW the graph
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=11)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{i}" for i in ids])
    ax3.set_xlabel("Experiment Number", fontsize=11)

    # --- Layout & Titles ---
    # Moved Title UP (0.98) and Subtitle DOWN (0.93) to separate them.
    plt.suptitle("Distribution Shift Study", fontsize=26, fontweight='bold', y=0.98)
    
    subtitle_str = f"{pretty_arch} | {dims} Dimensions | Sample Size: {sample_size}"
    plt.figtext(0.5, 0.93, subtitle_str, ha='center', fontsize=18, fontstyle='italic', color='#333333')

    # Architecture Footer
    fig.text(0.5, 0.02, f"Network Architecture: {arch_str}", ha='center', fontsize=14, fontweight='bold', 
             bbox=dict(facecolor='#FBFCFC', alpha=0.9, edgecolor='#AEB6BF', boxstyle='round,pad=1'))

    # Rect Top increased to 0.91 to bring graphs closer to the header block
    # Bottom adjusted to 0.15 to make room for the legend below Ax3
    plt.tight_layout(rect=[0.02, 0.10, 0.98, 0.91])
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / f"{folder_name}_summary.png", dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='figures')
    parser.add_argument('--all-dirs', action='store_true')
    args = parser.parse_args()

    for root, dirs, files in os.walk(args.base_path):
        if any(f.endswith('.log') for f in files):
            log_dir = Path(root)
            data = load_folder_data(log_dir)
            if data:
                folder_id = str(log_dir.relative_to(args.base_path)).replace('/', '_').replace('\\', '_')
                if not folder_id or folder_id == ".": folder_id = log_dir.name
                plot_comprehensive_analysis(data, args.output_dir, folder_id)
                print(f"✓ Generated: {folder_id}")

if __name__ == "__main__":
    main()
