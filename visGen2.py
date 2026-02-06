#!/usr/bin/env python3
"""
Visualize Mixed Shift Experiment Results (Generalized - Enhanced)
Reads from .log files in specified directories and generates graphs in multiple formats
EACH DIRECTORY GETS ITS OWN SEPARATE GRAPH - SEARCHES RECURSIVELY
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import time
import os

# Set style
plt.rcParams['figure.figsize'] = (15, 10)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def parse_log_file(log_path):
    """Parse a single log file to extract experiment metrics"""
    with open(log_path, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    # Extract autoencoder dimensions from features loaded line
    ae_dim_match = re.search(r'features loaded\. Shape = \(\d+, (\d+)\)', content)
    if ae_dim_match:
        metrics['autoencoder_dim'] = int(ae_dim_match.group(1))
    
    # Extract source calibration samples (from CULane features loaded line)
    src_calib_match = re.search(r'CULane features loaded\. Shape = \((\d+), \d+\)', content)
    if src_calib_match:
        metrics['src_calib'] = int(src_calib_match.group(1))
    else:
        # Alternative: look for the CULane samples line before features loaded
        src_calib_alt = re.search(r'CULane\).*?\((\d+) samples\).*?features loaded', content, re.DOTALL)
        if src_calib_alt:
            metrics['src_calib'] = int(src_calib_alt.group(1))
    
    # Extract target calibration samples (from "Total combined samples" line)
    tgt_calib_match = re.search(r'Total combined samples: (\d+)', content)
    if tgt_calib_match:
        metrics['tgt_calib'] = int(tgt_calib_match.group(1))
    
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
    
    # Extract source and target samples (for the test)
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

def load_experiment_data(logs_dir, log_pattern='*.log'):
    """Load all log files matching the pattern FROM A SINGLE DIRECTORY ONLY (not subdirectories)"""
    data = {}
    logs_path = Path(logs_dir)
    
    if not logs_path.exists():
        return data, [f"Directory not found: {logs_dir}"]
    
    errors = []
    
    # CRITICAL: Only get .log files directly in THIS directory, not subdirectories
    # Use iterdir() to avoid recursion
    log_files = sorted([f for f in logs_path.iterdir() if f.is_file() and f.suffix == '.log'])
    
    if not log_files:
        return data, [f"No log files found in {logs_dir}"]
    
    print(f"\nFound {len(log_files)} log files in {logs_dir}:")
    for f in log_files:
        print(f"  - {f.name}")
    
    print(f"\nProcessing files from {logs_dir}:")
    
    for log_file in log_files:
        numbers = re.findall(r'(\d+)', log_file.stem)
        if numbers:
            exp_num = int(numbers[-1])
        else:
            exp_num = len(data) + 1
        
        print(f"[Exp {exp_num:2d}] ", end='', flush=True)
        
        try:
            metrics = parse_log_file(log_file)
            if metrics:
                src = metrics.get('src_samples', None)
                tgt = metrics.get('tgt_samples', None)
                mmd = metrics.get('test_avg_mmd', None)
                tpr = metrics.get('tpr', None)
                ae_dim = metrics.get('autoencoder_dim', None)
                src_calib = metrics.get('src_calib', None)
                tgt_calib = metrics.get('tgt_calib', None)
                
                if src is not None and tgt is not None and mmd is not None:
                    data[exp_num] = metrics
                    print(f"✓ (AE_Dim={ae_dim}, Src_Calib={src_calib}, Tgt_Calib={tgt_calib}, Src={src}, Tgt={tgt}, TPR={tpr}%)")
                else:
                    error_msg = f"{log_file.name}: Missing required data (src={src}, tgt={tgt}, mmd={mmd})"
                    errors.append(error_msg)
                    print(f"⚠ MISSING: src={src}, tgt={tgt}, mmd={mmd}")
            else:
                error_msg = f"{log_file.name}: Failed to parse"
                errors.append(error_msg)
                print(f"✗ Failed to parse")
        except Exception as e:
            error_msg = f"{log_file.name}: Exception - {str(e)}"
            errors.append(error_msg)
            print(f"✗ Exception: {str(e)}")
    
    return data, errors

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
        'ratio': [],
        'autoencoder_dim': [],
        'src_calib': [],
        'tgt_calib': []
    }
    
    for exp_num, exp_data in sorted(data.items()):
        src = exp_data.get('src_samples', 0)
        tgt = exp_data.get('tgt_samples', 0)
        
        if 'test_avg_mmd' not in exp_data:
            continue
        
        if src > 0:
            ratio = tgt / src
        else:
            ratio = float('inf') if tgt > 0 else 0
        
        metrics['experiment_num'].append(exp_num)
        metrics['test_tpr'].append(exp_data.get('tpr', 0))
        metrics['test_avg_mmd'].append(exp_data.get('test_avg_mmd', 0))
        metrics['test_std_mmd'].append(exp_data.get('test_std_mmd', 0))
        metrics['src_samples'].append(src)
        metrics['tgt_samples'].append(tgt)
        metrics['tau'].append(exp_data.get('tau', 0))
        metrics['ratio'].append(ratio)
        metrics['autoencoder_dim'].append(exp_data.get('autoencoder_dim', 0))
        metrics['src_calib'].append(exp_data.get('src_calib', 0))
        metrics['tgt_calib'].append(exp_data.get('tgt_calib', 0))
    
    return metrics

def create_output_directories(base_dir='figures'):
    """Create output directories for each image format"""
    base_path = Path(base_dir)
    formats = ['svg', 'png', 'jpeg', 'pdf', 'eps']
    
    for fmt in formats:
        format_path = base_path / fmt
        format_path.mkdir(parents=True, exist_ok=True)
    
    return base_path

def generate_filename(ae_dim, src_calib, tgt_calib, n_experiments):
    """Generate a descriptive filename based on experiment parameters"""
    return f"mmd_analysis_dim{ae_dim}_src{src_calib}_tgt{tgt_calib}_n{n_experiments}"

def plot_comprehensive_analysis(metrics, output_dir='figures', folder_name='experiment'):
    """Generate comprehensive visualization of all experiments in multiple formats"""
    base_path = create_output_directories(output_dir)
    
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
    sorted_ae_dim = [metrics['autoencoder_dim'][i] for i in sorted_indices]
    sorted_src_calib = [metrics['src_calib'][i] for i in sorted_indices]
    sorted_tgt_calib = [metrics['tgt_calib'][i] for i in sorted_indices]
    
    # Get the most common values (mode)
    ae_dim_mode = max(set(sorted_ae_dim), key=sorted_ae_dim.count) if sorted_ae_dim else 0
    src_calib_mode = max(set(sorted_src_calib), key=sorted_src_calib.count) if sorted_src_calib else 0
    tgt_calib_mode = max(set(sorted_tgt_calib), key=sorted_tgt_calib.count) if sorted_tgt_calib else 0
    
    # VALIDATION: Check if all experiments have consistent calibration values
    unique_src_calib = set(sorted_src_calib)
    unique_tgt_calib = set(sorted_tgt_calib)
    unique_ae_dim = set(sorted_ae_dim)
    
    warnings = []
    if len(unique_src_calib) > 1 or len(unique_tgt_calib) > 1 or len(unique_ae_dim) > 1:
        warnings.append(f"Inconsistent values in {folder_name}")
        warnings.append(f"  Source calibration values: {unique_src_calib}")
        warnings.append(f"  Target calibration values: {unique_tgt_calib}")
        warnings.append(f"  Autoencoder dimensions: {unique_ae_dim}")
        print(f"\n⚠️  WARNING: Inconsistent values detected in {folder_name}!")
        print(f"   Source calibration values: {unique_src_calib}")
        print(f"   Target calibration values: {unique_tgt_calib}")
        print(f"   Autoencoder dimensions: {unique_ae_dim}")
        print(f"   This suggests data from multiple experiment types are mixed!")
    
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
    
    # Scientific title with sample information
    plt.suptitle(
        f'Maximum Mean Discrepancy Analysis of Distribution Shift Detection\n' +
        f'Feature Space Dimensionality: {ae_dim_mode} | ' +
        f'Source Samples: {src_calib_mode} | ' +
        f'Target Samples: {tgt_calib_mode} | ' +
        f'N={n_experiments} Experiments', 
        fontsize=16, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Generate descriptive filename
    base_filename = generate_filename(ae_dim_mode, src_calib_mode, tgt_calib_mode, n_experiments)
    
    # Save in multiple formats
    formats = {
        'svg': {'dpi': None},
        'png': {'dpi': 300},
        'jpeg': {'dpi': 300},
        'pdf': {'dpi': None},
        'eps': {'dpi': None}
    }
    
    for fmt, params in formats.items():
        output_filename = f'{base_filename}.{fmt}'
        output_file = base_path / fmt / output_filename
        
        if params['dpi']:
            plt.savefig(output_file, dpi=params['dpi'], bbox_inches='tight', format=fmt)
        else:
            plt.savefig(output_file, bbox_inches='tight', format=fmt)
        
        print(f"✓ Saved {fmt.upper()}: {output_file}")
    
    print(f"\nMapping (Sequential → Experiment) for {folder_name}:")
    for i, actual_exp in enumerate(sorted_exp_num):
        if sorted_ratio[i] == float('inf'):
            ratio_str = '∞'
        else:
            ratio_str = f'{sorted_ratio[i]:.2f}'
        print(f"  {i+1:2d} → Exp{actual_exp:2d} (AE_Dim: {sorted_ae_dim[i]}, Src_Calib: {sorted_src_calib[i]}, Tgt_Calib: {sorted_tgt_calib[i]}, Src: {sorted_src[i]}, Tgt: {sorted_tgt[i]}, Ratio: {ratio_str})")
    
    plt.close()  # Close the figure instead of showing it when batch processing
    
    return warnings

def should_exclude_directory(dir_path, base_path, exclude_patterns):
    """Check if a directory should be excluded based on exclusion patterns"""
    dir_path = Path(dir_path)
    base_path = Path(base_path)
    
    try:
        relative_path = dir_path.relative_to(base_path)
    except ValueError:
        # If not relative to base_path, check absolute path
        relative_path = dir_path
    
    for pattern in exclude_patterns:
        pattern_path = Path(pattern)
        # Check if the directory matches the exclusion pattern or is a subdirectory of it
        if relative_path == pattern_path or pattern_path in relative_path.parents or str(pattern_path) in str(relative_path):
            return True
    
    return False

def find_log_directories(base_path='LocalBash', exclude_dirs=None):
    """Find all directories (RECURSIVELY) in LocalBash that contain log files, excluding specified directories"""
    base_path = Path(base_path)
    
    if exclude_dirs is None:
        exclude_dirs = []
    
    # Normalize exclude patterns to be relative to base_path
    exclude_patterns = []
    for exclude_dir in exclude_dirs:
        exclude_path = Path(exclude_dir)
        try:
            # Try to make it relative to base_path
            if exclude_path.is_absolute():
                exclude_patterns.append(exclude_path.relative_to(base_path))
            else:
                exclude_patterns.append(exclude_path)
        except ValueError:
            # If it fails, just use the pattern as-is
            exclude_patterns.append(exclude_path)
    
    if not base_path.exists():
        print(f"❌ Base path not found: {base_path}")
        return []
    
    log_dirs = []
    excluded_dirs = []
    
    # Walk through all subdirectories RECURSIVELY
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        
        # Check if this directory should be excluded
        if should_exclude_directory(root_path, base_path, exclude_patterns):
            excluded_dirs.append(root_path)
            # Don't descend into subdirectories of excluded directories
            dirs[:] = []
            continue
        
        # Check if current directory has any .log files directly in it
        log_files = [f for f in files if f.endswith('.log')]
        if log_files:
            log_dirs.append(root_path)
            try:
                rel_path = root_path.relative_to(base_path)
                print(f"  Found directory with logs: {rel_path}")
            except ValueError:
                print(f"  Found directory with logs: {root_path}")
    
    if excluded_dirs:
        print(f"\n  Excluded {len(excluded_dirs)} directories:")
        for excluded in excluded_dirs[:5]:  # Show first 5
            try:
                rel_path = excluded.relative_to(base_path)
                print(f"    - {rel_path}")
            except ValueError:
                print(f"    - {excluded}")
        if len(excluded_dirs) > 5:
            print(f"    ... and {len(excluded_dirs) - 5} more")
    
    return sorted(log_dirs)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Visualize experiment results from log files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single directory
  python visualizeGeneralized.py --log-path LocalBash/micro
  
  # Process all directories in LocalBash RECURSIVELY (each gets its own graph)
  python visualizeGeneralized.py --all-dirs --base-path LocalBash
  
  # Process with custom output directory
  python visualizeGeneralized.py --all-dirs --base-path LocalBash --output-dir results
        """
    )
    
    parser.add_argument(
        '--log-path',
        type=str,
        help='Path to directory containing log files'
    )
    
    parser.add_argument(
        '--all-dirs',
        action='store_true',
        help='Process all directories in base-path that contain log files RECURSIVELY (each directory gets its own graph)'
    )
    
    parser.add_argument(
        '--base-path',
        type=str,
        default='LocalBash',
        help='Base path to search for log directories (default: LocalBash)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures',
        help='Directory to save output figures (default: figures)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.log',
        help='Glob pattern for log files (default: *.log)'
    )
    
    args = parser.parse_args()
    
    if not args.log_path and not args.all_dirs:
        parser.error("Either --log-path or --all-dirs must be specified")
    
    # Directories to exclude
    exclude_dirs = ['CCE_IncorrectLog']
    
    # Track all errors and warnings
    all_errors = {}
    all_warnings = {}
    skipped_dirs = []
    
    # Determine which directories to process
    if args.all_dirs:
        print(f"\n{'='*60}")
        print(f"Searching RECURSIVELY in {args.base_path} for directories with .log files...")
        print(f"Excluding: {', '.join(exclude_dirs)}")
        print(f"{'='*60}")
        log_directories = find_log_directories(args.base_path, exclude_dirs=exclude_dirs)
        if not log_directories:
            print(f"❌ No directories with log files found in {args.base_path}")
            return
        print(f"\n{'='*60}")
        print(f"Found {len(log_directories)} directories with log files")
        print(f"{'='*60}")
    else:
        log_directories = [Path(args.log_path)]
    
    # Process each directory SEPARATELY
    successful_graphs = 0
    for log_dir in log_directories:
        # Get relative path for better display
        try:
            folder_display = log_dir.relative_to(Path(args.base_path)) if args.all_dirs else log_dir
        except:
            folder_display = log_dir
        
        folder_name = str(folder_display).replace('/', '_').replace('\\', '_')
        
        print("\n" + "="*60)
        print(f"PROCESSING: {folder_display}")
        print("="*60)
        print(f"Full path: {log_dir}")
        print(f"Output dir: {args.output_dir}")
        
        # Load data ONLY from this specific directory (not its subdirectories)
        data, errors = load_experiment_data(log_dir, args.pattern)
        
        if errors:
            all_errors[str(folder_display)] = errors
        
        if not data:
            print(f"\n❌ No valid data found in {folder_display}!")
            skipped_dirs.append(str(folder_display))
            continue
        
        print(f"\n✓ Successfully loaded {len(data)} experiments from {folder_display}")
        
        # Extract metrics from THIS directory only
        metrics = extract_metrics(data)
        
        if not metrics['experiment_num']:
            print(f"\n❌ No valid metrics in {folder_display}!")
            skipped_dirs.append(str(folder_display))
            continue
        
        print(f"✓ Processing {len(metrics['experiment_num'])} experiments from {folder_display}\n")
        
        # Generate graph for THIS directory only
        warnings = plot_comprehensive_analysis(
            metrics, 
            output_dir=args.output_dir,
            folder_name=folder_name
        )
        
        if warnings:
            all_warnings[str(folder_display)] = warnings
        
        successful_graphs += 1
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Successfully generated {successful_graphs} graphs")
    
    if skipped_dirs:
        print(f"\n⚠️  Skipped {len(skipped_dirs)} directories (no valid data):")
        for skipped in skipped_dirs:
            print(f"   - {skipped}")
    
    if all_warnings:
        print(f"\n⚠️  Warnings ({len(all_warnings)} directories with issues):")
        for dir_name, warnings in all_warnings.items():
            print(f"\n   {dir_name}:")
            for warning in warnings:
                print(f"      {warning}")
    
    if all_errors:
        print(f"\n❌ Errors ({len(all_errors)} directories with errors):")
        for dir_name, errors in all_errors.items():
            print(f"\n   {dir_name}:")
            for error in errors:
                print(f"      {error}")
    
    if not all_warnings and not all_errors and not skipped_dirs:
        print("\n✅ No errors or warnings!")
    
    print("\n" + "="*60)
    print(f"✓ All processing complete!")
    print("="*60)

if __name__ == "__main__":
    main()
    