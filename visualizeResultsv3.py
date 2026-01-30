#!/usr/bin/env python3
"""
Visualize Mixed Shift Experiment Results (Micro)
Reads from .log files in LocalBash/micro directory
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.rcParams['figure.figsize'] = (15, 10)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def parse_log_file(log_path):
    """Parse a single log file to extract experiment metrics"""
    with open(log_path, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    # Extract Tau
    tau_match = re.search(r'\[RESULT\] τ\([\d.]+\) = ([\d.]+)', content)
    if tau_match:
        metrics['tau'] = float(tau_match.group(1))
    
    # Extract calibration MMD
    calib_match = re.search(r'Mean MMD \(same-distribution\): ([\d.]+) ± ([\d.]+)', content)
    if calib_match:
        metrics['calib_mmd'] = float(calib_match.group(1))
        metrics['calib_std'] = float(calib_match.group(2))
    
    # Extract sanity check MMD
    sanity_match = re.search(r'\[SANITY CHECK\] MMD.*? = ([\d.]+)', content)
    if sanity_match:
        metrics['sanity_mmd'] = float(sanity_match.group(1))
    
    # Extract test results
    test_avg_match = re.search(r'Average MMD: ([\d.]+) ± ([\d.]+)', content)
    if test_avg_match:
        metrics['test_avg_mmd'] = float(test_avg_match.group(1))
        metrics['test_std_mmd'] = float(test_avg_match.group(2))
    
    # Extract TPR
    tpr_match = re.search(r'TPR \(true positive rate\) over \d+ runs: ([\d.]+)%', content)
    if tpr_match:
        metrics['tpr'] = float(tpr_match.group(1))
    
    # Extract source and target samples - look for the Mixed Dataloader info in STEP 3
    # Pattern 1: Try to find in the first [RUN] section after STEP 3
    run_section = re.search(r'\[STEP 3\].*?\[RUN 1\]', content, re.DOTALL)
    if run_section:
        section_text = run_section.group(0)
        
        # Look for CULane samples
        src_matches = re.findall(r'CULane.*?\((\d+) samples\)', section_text)
        if src_matches:
            metrics['src_samples'] = int(src_matches[-1])
        
        # Look for Curvelanes samples  
        tgt_matches = re.findall(r'Curvelanes.*?\((\d+) samples\)', section_text)
        if tgt_matches:
            metrics['tgt_samples'] = int(tgt_matches[-1])
    
    # Alternative: Look for the data shift test description line
    if 'src_samples' not in metrics or 'tgt_samples' not in metrics:
        shift_desc = re.search(r'\[STEP 3\] Data Shift Test:.*?\((\d+)\).*?Curvelanes.*?\((\d+)\).*?CULane', content)
        if shift_desc:
            metrics['tgt_samples'] = int(shift_desc.group(1))
            metrics['src_samples'] = int(shift_desc.group(2))
    
    return metrics if metrics else None

def load_experiment_data(logs_dir='LocalBash/micro'):
    """Load all MCE*_micro.log files"""
    data = {}
    logs_path = Path(logs_dir)
    
    log_files = sorted(logs_path.glob('MCE*.log'), key=lambda x: int(re.search(r'(\d+)', x.stem).group(1)))
    
    print(f"\nFound {len(log_files)} MCE*.log files:")
    for f in log_files:
        print(f"  - {f.name}")
    
    print(f"\nProcessing files:")
    
    for log_file in log_files:
        exp_match = re.search(r'MCE(\d+)\.log', log_file.name)
        if exp_match:
            exp_num = int(exp_match.group(1))
            print(f"[MCE{exp_num:2d}] ", end='', flush=True)
            metrics = parse_log_file(log_file)
            if metrics:
                src = metrics.get('src_samples', None)
                tgt = metrics.get('tgt_samples', None)
                mmd = metrics.get('test_avg_mmd', None)
                tpr = metrics.get('tpr', None)
                
                # Allow src=0 now (for cases like MCE11)
                if src is not None and tgt is not None and mmd is not None:
                    data[exp_num] = metrics
                    print(f"✓ (Src={src}, Tgt={tgt}, TPR={tpr}%)")
                else:
                    print(f"⚠ MISSING: src={src}, tgt={tgt}, mmd={mmd}")
            else:
                print(f"✗ Failed to parse")
    
    return data

def extract_metrics(data):
    """Convert parsed data into metrics arrays"""
    metrics = {
        'experiment_num': [],
        'test_tpr': [],
        'test_avg_mmd': [],
        'test_std_mmd': [],
        'src_samples': [],
        'tgt_samples': [],
        'tau': [],
        'ratio': []
    }
    
    for exp_num, exp_data in sorted(data.items()):
        src = exp_data.get('src_samples', 0)
        tgt = exp_data.get('tgt_samples', 0)
        
        # Don't skip if src == 0, just check for valid MMD data
        if 'test_avg_mmd' not in exp_data:
            continue
        
        # Handle division by zero for ratio calculation
        if src > 0:
            ratio = tgt / src
        else:
            # If src is 0, we'll use a very large ratio (or special handling)
            ratio = float('inf') if tgt > 0 else 0
        
        metrics['experiment_num'].append(exp_num)
        metrics['test_tpr'].append(exp_data.get('tpr', 0))
        metrics['test_avg_mmd'].append(exp_data.get('test_avg_mmd', 0))
        metrics['test_std_mmd'].append(exp_data.get('test_std_mmd', 0))
        metrics['src_samples'].append(src)
        metrics['tgt_samples'].append(tgt)
        metrics['tau'].append(exp_data.get('tau', 0))
        metrics['ratio'].append(ratio)
    
    return metrics

def plot_comprehensive_analysis(metrics, output_dir='figures'):
    """Generate comprehensive visualization of all experiments"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    sorted_indices = np.argsort(metrics['ratio'])
    n_experiments = len(metrics['experiment_num'])
    x_positions = np.arange(n_experiments)
    
    sorted_exp_num = [metrics['experiment_num'][i] for i in sorted_indices]
    sorted_ratio = [metrics['ratio'][i] for i in sorted_indices]
    sorted_tpr = [metrics['test_tpr'][i] for i in sorted_indices]
    sorted_avg_mmd = [metrics['test_avg_mmd'][i] for i in sorted_indices]
    sorted_tau = [metrics['tau'][i] for i in sorted_indices]
    sorted_src = [metrics['src_samples'][i] for i in sorted_indices]
    sorted_tgt = [metrics['tgt_samples'][i] for i in sorted_indices]
    
    sequential_labels = [f"{i+1}" for i in range(n_experiments)]
    
    fig = plt.figure(figsize=(18, 7))
    
    # 1. TPR
    ax1 = plt.subplot(1, 3, 1)
    colors = ['#06A77D' if tpr == 100 else '#D62828' for tpr in sorted_tpr]
    bars = plt.bar(x_positions, sorted_tpr, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for i, (bar, tpr) in enumerate(zip(bars, sorted_tpr)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{tpr:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.xlabel('Experiment Number (Ordered by Ratio)', fontsize=13, fontweight='bold')
    plt.ylabel('TPR (%)', fontsize=14, fontweight='bold')
    plt.title('True Positive Rate', fontsize=16, fontweight='bold', pad=15)
    plt.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Perfect Detection')
    plt.xticks(x_positions, sequential_labels, fontsize=10)
    plt.ylim([0, 110])
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=11)
    
    # 2. MMD
    ax2 = plt.subplot(1, 3, 2)
    plt.plot(x_positions, sorted_avg_mmd, 'o-', linewidth=2.5, markersize=10, color='#F18F01', label='Average MMD')
    plt.plot(x_positions, sorted_tau, 's--', linewidth=2, markersize=8, color='#2E86AB', alpha=0.7, label='Tau (Threshold)')
    
    plt.xlabel('Experiment Number (Ordered by Ratio)', fontsize=13, fontweight='bold')
    plt.ylabel('MMD Value', fontsize=14, fontweight='bold')
    plt.title('Average MMD vs Sample Ratio', fontsize=16, fontweight='bold', pad=15)
    plt.xticks(x_positions, sequential_labels, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='best')
    
    # 3. Samples
    ax3 = plt.subplot(1, 3, 3)
    width = 0.25
    bars1 = plt.bar(x_positions - width/2, sorted_src, width, label='Source Samples', alpha=0.8, color='#06A77D', edgecolor='black')
    bars2 = plt.bar(x_positions + width/2, sorted_tgt, width, label='Target Samples', alpha=0.8, color='#F18F01', edgecolor='black')
    
    max_height = max(max(sorted_src), max(sorted_tgt)) if sorted_src and sorted_tgt else 1
    
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        plt.text(bar1.get_x() + bar1.get_width()/2., height1 + max_height * 0.015,
                f'{int(sorted_src[i])}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#06A77D')
        
        plt.text(bar2.get_x() + bar2.get_width()/2., height2 + max_height * 0.015,
                f'{int(sorted_tgt[i])}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#F18F01')
    
    # Display ratio (handle infinity case)
    for i in range(len(sorted_exp_num)):
        if sorted_ratio[i] == float('inf'):
            ratio_text = '∞'
        else:
            ratio_text = f'{sorted_ratio[i]:.2f}'
        plt.text(x_positions[i], -max_height * 0.08, ratio_text, 
                ha='center', va='top', fontsize=8, style='italic', color='#555')
    
    plt.xlabel('Experiment Number (Ordered by Ratio)\n(Ratio shown below)', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=14, fontweight='bold')
    plt.title('Source vs Target Samples', fontsize=16, fontweight='bold', pad=15)
    plt.xticks(x_positions, sequential_labels, fontsize=10)
    plt.xlim([-0.7, n_experiments - 0.3])
    plt.ylim([-max_height * 0.12, max_height * 1.08])
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Mixed Shift Experiment Results (Micro): {n_experiments} Experiments (Ordered by Tgt/Src Ratio)', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_file = output_path / 'mixed_shift_experiments_micro_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {output_file}")
    
    print("\nMapping (Sequential → MCE):")
    for i, actual_exp in enumerate(sorted_exp_num):
        if sorted_ratio[i] == float('inf'):
            ratio_str = '∞'
        else:
            ratio_str = f'{sorted_ratio[i]:.2f}'
        print(f"  {i+1:2d} → MCE{actual_exp:2d} (ratio: {ratio_str})")
    
    plt.show()

def main():
    """Main execution function"""
    print("="*60)
    print("MIXED SHIFT EXPERIMENT VISUALIZER (MICRO)")
    print("="*60)
    
    data = load_experiment_data('LocalBash/micro')
    
    if not data:
        print("\n❌ No data found!")
        return
    
    print(f"\n✓ Successfully loaded {len(data)} experiments: MCE{sorted(data.keys())}")
    
    metrics = extract_metrics(data)
    
    if not metrics['experiment_num']:
        print("\n❌ No valid metrics!")
        return
    
    print(f"✓ Processing {len(metrics['experiment_num'])} experiments\n")
    
    plot_comprehensive_analysis(metrics)
    
    print("\n" + "="*60)
    print("✓ Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
