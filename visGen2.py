#!/usr/bin/env python3
"""
Visualize Mixed Shift Experiment Results (Robust Version)
Handles mixed/misplaced log files by grouping them based on their actual configuration
extracted from filename and log content.
"""

import re
import os
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style
plt.rcParams['figure.figsize'] = (15, 10)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Architecture configurations
ARCHITECTURE_CONFIGS = {
    "d128rel": {
        "name": "Remove Extra Layer",
        "layers": [4096, 1024, 128]
    },
    "d64rel": {
        "name": "Remove Extra Layer",
        "layers": [4096, 1024, 64]
    },
    "d32rel": {
        "name": "Remove Extra Layer",
        "layers": [4096, 1024, 32]
    },
    "d128gdd": {
        "name": "Gradually Decrease Dimensions",
        "layers": [4096, 512, 128]
    },
    "d64gdd": {
        "name": "Gradually Decrease Dimensions",
        "layers": [4096, 512, 64]
    },
    "d32gdd": {
        "name": "Gradually Decrease Dimensions",
        "layers": [4096, 512, 32]
    },
    "d64ids": {
        "name": "Increase Dimension Size",
        "layers": [4096, 1024, 256, 64]
    },
    "d128ids": {
        "name": "Increase Dimension Size",
        "layers": [4096, 1024, 256, 128]
    },
    "d32ids": {
        "name": "Increase Dimension Size",
        "layers": [4096, 1024, 256, 32]
    },
    "d32": {
        "name": "32D Standard",
        "layers": [4096, 1024, 256, 32]
    },
    "orig": {
        "name": "Original Architecture",
        "layers": [4096, 1024, 256]
    }
}

def extract_config_from_filename(filename):
    """
    Extract configuration from filename pattern.
    
    Patterns:
    - d64ids = 64 dimensions, increase dimension size
    - d128gdd = 128 dimensions, gradually decrease dimensions
    - d32rel = 32 dimensions, remove extra layer
    - K100 = 100 samples, K1000 = 1000 samples, no K = 10 samples
    
    Examples:
    - d64idsK100.log
    - d128gdd10.log (10 samples, no K)
    - 5d64gddK1000.log
    """
    stem = Path(filename).stem
    
    config = {
        'dim_config': 'unknown',
        'calibration_samples': 'unknown',
        'experiment_num': None
    }
    
    # Extract dimension configuration (d64ids, d128gdd, d32rel, etc.)
    # Pattern: look for dXXXyyy where XXX is digits and yyy is letters
    dim_match = re.search(r'd(\d+)(ids|gdd|rel)', stem, re.IGNORECASE)
    if dim_match:
        dim_size = dim_match.group(1)
        dim_type = dim_match.group(2).lower()
        config['dim_config'] = f"d{dim_size}{dim_type}"
    
    # Extract calibration samples (K100, K1000, or just digits)
    # First try K pattern
    k_match = re.search(r'K(\d+)', stem)
    if k_match:
        config['calibration_samples'] = int(k_match.group(1))
    else:
        # Try to find just digits at the end (for 10 sample case)
        # Look for pattern like "d64gdd10" or "5d64gdd10"
        digit_match = re.search(r'd\d+(?:ids|gdd|rel)(\d+)', stem)
        if digit_match:
            config['calibration_samples'] = int(digit_match.group(1))
    
    # Extract experiment number (usually at the start or in the middle)
    # Patterns: "5d64gdd" or "d64ids5" or just "5" somewhere
    exp_nums = re.findall(r'(\d+)', stem)
    if exp_nums:
        # Take the first number that's not part of dimension or K value
        for num in exp_nums:
            num_int = int(num)
            # Skip if it's likely a dimension size (32, 64, 128) or calibration (10, 100, 1000)
            if num_int not in [10, 32, 64, 100, 128, 1000]:
                config['experiment_num'] = num_int
                break
        
        # If we didn't find it, take the first number
        if config['experiment_num'] is None and exp_nums:
            config['experiment_num'] = int(exp_nums[0])
    
    return config

def parse_log_file(log_path):
    """Parse a single log file to extract experiment metrics"""
    try:
        # Try UTF-8 first
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            # Try latin-1 as fallback
            with open(log_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                # Try cp1252 (Windows encoding)
                with open(log_path, 'r', encoding='cp1252') as f:
                    content = f.read()
            except Exception:
                # Last resort: read with errors='ignore'
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
    
    metrics = {}
    
    # Extract experiment info from the header
    # "Experiment 1: Src=0% (0 samples), Tgt=100% (10 samples)"
    exp_header = re.search(r'Experiment\s+(\d+):\s+Src=[\d.]+%\s+\((\d+)\s+samples?\),\s+Tgt=[\d.]+%\s+\((\d+)\s+samples?\)', content)
    if exp_header:
        metrics['experiment_num'] = int(exp_header.group(1))
        metrics['src_samples'] = int(exp_header.group(2))
        metrics['tgt_samples'] = int(exp_header.group(3))
    
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
    
    return metrics if metrics else None

def get_config_description(dim_config):
    """Get human-readable description of dimension config"""
    if not dim_config or dim_config == 'orig':
        return "Original (256D)"
    
    # Extract dimension and type
    match = re.match(r'd(\d+)(.+)?', dim_config)
    if match:
        dim = match.group(1)
        config_type = match.group(2) if match.group(2) else ""
        
        # Simple mapping for common types
        type_mapping = {
            "rel": "Remove Extra Layer",
            "ids": "Increase Dimension Size",
            "gdd": "Gradually Decrease Dimensions",
            "": "Standard"
        }
        type_name = type_mapping.get(config_type, config_type.upper() if config_type else "Standard")
        return f"{dim}D - {type_name}"
    
    return dim_config

def get_architecture_text(dim_config):
    """Get architecture text for display at bottom of graph"""
    if not dim_config or dim_config not in ARCHITECTURE_CONFIGS:
        # Default/unknown config
        if dim_config == 'orig' or not dim_config:
            return "Original Architecture\n4096 → 1024 → 256"
        return f"Architecture: {dim_config}\n(Configuration details not available)"
    
    config = ARCHITECTURE_CONFIGS[dim_config]
    name = config['name']
    layers = config['layers']
    layer_text = " → ".join(str(layer) for layer in layers)
    
    return f"{name}\n{layer_text}"

def create_config_key(file_config, log_metrics):
    """Create a unique key for grouping experiments by configuration"""
    dim_config = file_config.get('dim_config', 'unknown')
    src_calib = log_metrics.get('src_calib', file_config.get('calibration_samples', 'unknown'))
    tgt_calib = log_metrics.get('tgt_calib', file_config.get('calibration_samples', 'unknown'))
    ae_dim = log_metrics.get('autoencoder_dim', 'unknown')
    
    return f"{dim_config}__ae{ae_dim}_src{src_calib}_tgt{tgt_calib}"

def load_experiment_data_grouped(logs_dirs, log_pattern='*.log', recursive=True):
    """
    Load all log files from multiple directories and group them by their actual configuration.
    
    Args:
        logs_dirs: List of directory paths to search for log files
        log_pattern: Pattern to match log files
        recursive: Whether to search recursively
    
    Returns:
        Dictionary where keys are configuration identifiers and values are experiment data.
    """
    if isinstance(logs_dirs, (str, Path)):
        logs_dirs = [logs_dirs]
    
    errors = []
    grouped_data = defaultdict(lambda: {'experiments': {}, 'metadata': {}, 'source_dirs': set()})
    
    total_files = 0
    all_log_files = []
    
    # Collect all log files from all directories
    for logs_dir in logs_dirs:
        logs_path = Path(logs_dir)
        
        if not logs_path.exists():
            errors.append(f"Directory not found: {logs_dir}")
            print(f"⚠ Directory not found: {logs_dir}")
            continue
        
        # Get log files (recursively or not)
        if recursive:
            log_files = list(logs_path.rglob('*.log'))
        else:
            log_files = [f for f in logs_path.iterdir() if f.is_file() and f.suffix == '.log']
        
        for log_file in log_files:
            all_log_files.append((log_file, logs_dir))
        
        print(f"Found {len(log_files)} log files in {logs_dir}")
    
    # Sort all log files by filename (not by directory)
    all_log_files.sort(key=lambda x: x[0].name)
    
    total_files = len(all_log_files)
    print(f"\nTotal files to process: {total_files}")
    print(f"\nProcessing files (sorted by filename):")
    
    for log_file, source_dir in all_log_files:
        # Extract configuration from filename
        file_config = extract_config_from_filename(log_file.name)
        
        # Get relative path for display
        try:
            rel_path = log_file.relative_to(Path(source_dir))
            display_path = str(rel_path)
        except ValueError:
            display_path = log_file.name
        
        print(f"[{log_file.name}] ", end='', flush=True)
        
        try:
            metrics = parse_log_file(log_file)
            
            if file_config['dim_config'] != 'unknown':
                print(f"[Config: {file_config['dim_config']}] ", end='', flush=True)
            
            if metrics:
                src = metrics.get('src_samples', None)
                tgt = metrics.get('tgt_samples', None)
                mmd = metrics.get('test_avg_mmd', None)
                tpr = metrics.get('tpr', None)
                ae_dim = metrics.get('autoencoder_dim', None)
                src_calib = metrics.get('src_calib', None)
                tgt_calib = metrics.get('tgt_calib', None)
                
                # Use experiment number from log if available, otherwise from filename
                exp_num = metrics.get('experiment_num', file_config.get('experiment_num', 0))
                
                if src is not None and tgt is not None and mmd is not None:
                    # Merge file config and metrics
                    metrics['file_config'] = file_config
                    metrics['dim_config'] = file_config['dim_config']
                    
                    # Create configuration key for grouping
                    config_key = create_config_key(file_config, metrics)
                    
                    # Store experiment in appropriate group
                    # Use filename as unique identifier
                    unique_key = log_file.stem
                    grouped_data[config_key]['experiments'][unique_key] = {
                        'metrics': metrics,
                        'file_path': log_file,
                        'exp_num': exp_num,
                        'source_dir': source_dir
                    }
                    
                    # Track source directory
                    grouped_data[config_key]['source_dirs'].add(str(Path(source_dir)))
                    
                    # Update metadata for this group
                    if not grouped_data[config_key]['metadata']:
                        grouped_data[config_key]['metadata'] = {
                            'ae_dim': ae_dim,
                            'src_calib': src_calib,
                            'tgt_calib': tgt_calib,
                            'dim_config': file_config['dim_config']
                        }
                    
                    print(f"✓ Group: {config_key} (Exp={exp_num}, Src={src}, Tgt={tgt}, TPR={tpr}%)")
                else:
                    error_msg = f"{display_path}: Missing required data (src={src}, tgt={tgt}, mmd={mmd})"
                    errors.append(error_msg)
                    print(f"⚠ MISSING: src={src}, tgt={tgt}, mmd={mmd}")
            else:
                error_msg = f"{display_path}: Failed to parse"
                errors.append(error_msg)
                print(f"✗ Failed to parse")
        except Exception as e:
            error_msg = f"{display_path}: Exception - {str(e)}"
            errors.append(error_msg)
            print(f"✗ Exception: {str(e)}")
    
    # Convert defaultdict to regular dict and summarize
    result = dict(grouped_data)
    
    # Convert sets to lists for JSON serialization
    for config_key in result:
        result[config_key]['source_dirs'] = list(result[config_key]['source_dirs'])
    
    print(f"\n{'='*60}")
    print(f"GROUPING SUMMARY (from {len(logs_dirs)} source directories, {total_files} total files):")
    print(f"{'='*60}")
    for config_key, group_data in result.items():
        n_experiments = len(group_data['experiments'])
        metadata = group_data['metadata']
        source_dirs = group_data['source_dirs']
        
        print(f"\nGroup: {config_key}")
        print(f"  Configuration: {metadata.get('dim_config', 'unknown')}")
        print(f"  AE Dimension: {metadata.get('ae_dim', 'unknown')}")
        print(f"  Source Calibration: {metadata.get('src_calib', 'unknown')}")
        print(f"  Target Calibration: {metadata.get('tgt_calib', 'unknown')}")
        print(f"  Number of experiments: {n_experiments}")
        print(f"  Source directories: {len(source_dirs)}")
        for src_dir in source_dirs:
            print(f"    - {src_dir}")
    
    return result, errors

def extract_metrics_from_group(group_data):
    """Convert grouped experiment data into metrics arrays"""
    experiments = group_data['experiments']
    
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
        'tgt_calib': [],
        'dim_config': [],
        'file_paths': [],
        'source_dirs': []
    }
    
    for unique_key, exp_data in sorted(experiments.items(), key=lambda x: x[1]['exp_num']):
        exp_metrics = exp_data['metrics']
        
        src = exp_metrics.get('src_samples', 0)
        tgt = exp_metrics.get('tgt_samples', 0)
        
        if 'test_avg_mmd' not in exp_metrics:
            continue
        
        if src > 0:
            ratio = tgt / src
        else:
            ratio = float('inf') if tgt > 0 else 0
        
        metrics['experiment_num'].append(exp_data['exp_num'])
        metrics['test_tpr'].append(exp_metrics.get('tpr', 0))
        metrics['test_avg_mmd'].append(exp_metrics.get('test_avg_mmd', 0))
        metrics['test_std_mmd'].append(exp_metrics.get('test_std_mmd', 0))
        metrics['src_samples'].append(src)
        metrics['tgt_samples'].append(tgt)
        metrics['tau'].append(exp_metrics.get('tau', 0))
        metrics['ratio'].append(ratio)
        metrics['autoencoder_dim'].append(exp_metrics.get('autoencoder_dim', 0))
        metrics['src_calib'].append(exp_metrics.get('src_calib', 0))
        metrics['tgt_calib'].append(exp_metrics.get('tgt_calib', 0))
        
        dim_config = exp_metrics.get('dim_config', 'unknown')
        metrics['dim_config'].append(dim_config)
        metrics['file_paths'].append(str(exp_data['file_path']))
        metrics['source_dirs'].append(exp_data.get('source_dir', 'unknown'))
    
    return metrics

def create_output_directories(base_dir='figures'):
    """Create output directories for each image format"""
    base_path = Path(base_dir)
    formats = ['svg', 'png', 'jpeg', 'pdf', 'eps']
    
    for fmt in formats:
        format_path = base_path / fmt
        format_path.mkdir(parents=True, exist_ok=True)
    
    return base_path

def generate_filename(ae_dim, src_calib, tgt_calib, n_experiments, dim_config=None):
    """Generate a descriptive filename based on experiment parameters"""
    base = f"mmd_analysis_dim{ae_dim}_src{src_calib}_tgt{tgt_calib}_n{n_experiments}"
    if dim_config and dim_config != 'unknown' and dim_config != 'orig':
        base += f"_{dim_config}"
    return base

def plot_comprehensive_analysis(metrics, output_dir='figures', group_name='experiment'):
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
    sorted_dim_config = [metrics['dim_config'][i] for i in sorted_indices]
    sorted_file_paths = [metrics['file_paths'][i] for i in sorted_indices]
    sorted_source_dirs = [metrics['source_dirs'][i] for i in sorted_indices]
    
    # Get the most common values (mode) - should all be the same in a group
    ae_dim_mode = sorted_ae_dim[0] if sorted_ae_dim else 0
    src_calib_mode = sorted_src_calib[0] if sorted_src_calib else 0
    tgt_calib_mode = sorted_tgt_calib[0] if sorted_tgt_calib else 0
    dim_config_mode = sorted_dim_config[0] if sorted_dim_config else 'unknown'
    
    sequential_labels = [f"{i+1}" for i in range(n_experiments)]
    
    # Create figure with extra space at bottom for architecture text
    fig = plt.figure(figsize=(18, 8))
    
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
    
    # Scientific title with configuration information
    config_desc = get_config_description(dim_config_mode)
    
    # Count unique source directories
    unique_sources = set(sorted_source_dirs)
    n_sources = len(unique_sources)
    source_note = f" (from {n_sources} source dir{'s' if n_sources > 1 else ''})" if n_sources > 1 else ""
    
    plt.suptitle(
        f'Maximum Mean Discrepancy Analysis of Distribution Shift Detection\n' +
        f'Architecture: {config_desc} | ' +
        f'Source Samples: {src_calib_mode} | ' +
        f'Target Samples: {tgt_calib_mode} | ' +
        f'N={n_experiments} Experiments{source_note}', 
        fontsize=16, fontweight='bold', y=0.97
    )
    
    # Add architecture text at the bottom
    arch_text = get_architecture_text(dim_config_mode)
    fig.text(0.5, 0.02, arch_text, 
             ha='center', va='bottom',
             fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    
    # Generate descriptive filename
    base_filename = generate_filename(ae_dim_mode, src_calib_mode, tgt_calib_mode, n_experiments, dim_config_mode)
    
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
    
    print(f"\nMapping (Sequential → Experiment) for {group_name}:")
    for i, actual_exp in enumerate(sorted_exp_num):
        if sorted_ratio[i] == float('inf'):
            ratio_str = '∞'
        else:
            ratio_str = f'{sorted_ratio[i]:.2f}'
        config_str = sorted_dim_config[i] if sorted_dim_config[i] != 'unknown' else 'N/A'
        file_path = Path(sorted_file_paths[i])
        source_dir = Path(sorted_source_dirs[i]).name if sorted_source_dirs[i] != 'unknown' else 'N/A'
        print(f"  {i+1:2d} → Exp{actual_exp:2d} [{file_path.name}] from [{source_dir}] (Config: {config_str}, Src: {sorted_src[i]}, Tgt: {sorted_tgt[i]}, Ratio: {ratio_str})")
    
    plt.close()

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Visualize experiment results from log files (handles mixed/misplaced files from multiple directories)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single directory (groups files by configuration)
  python visualize_mixed_shift_robust.py --log-path LocalBash/micro
  
  # Process multiple directories (merges files by configuration)
  python visualize_mixed_shift_robust.py --log-path LocalBash/RemoveExtraLayer LocalBash/GraduallyDecreaseDimensions
  
  # Process directory recursively
  python visualize_mixed_shift_robust.py --log-path LocalBash --recursive
  
  # Process with custom output directory
  python visualize_mixed_shift_robust.py --log-path LocalBash --recursive --output-dir results
        """
    )
    
    parser.add_argument(
        '--log-path',
        type=str,
        nargs='+',  # Accept multiple paths
        required=True,
        help='Path(s) to directory/directories containing log files (can specify multiple)'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search recursively for log files'
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
    
    print(f"\n{'='*60}")
    print(f"Processing log files from {len(args.log_path)} source director{'ies' if len(args.log_path) > 1 else 'y'}:")
    for path in args.log_path:
        print(f"  - {path}")
    print(f"Recursive: {args.recursive}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")
    
    # Load and group experiments by configuration
    grouped_data, errors = load_experiment_data_grouped(args.log_path, args.pattern, args.recursive)
    
    if errors:
        print(f"\n⚠️  Encountered {len(errors)} errors during parsing:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"   - {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more errors")
    
    if not grouped_data:
        print(f"\n❌ No valid experiment groups found!")
        return
    
    print(f"\n{'='*60}")
    print(f"Found {len(grouped_data)} distinct experiment configurations")
    print(f"{'='*60}")
    
    # Generate graphs for each group
    successful_graphs = 0
    for group_key, group_data in grouped_data.items():
        print(f"\n{'='*60}")
        print(f"GENERATING GRAPH FOR: {group_key}")
        print(f"{'='*60}")
        
        metrics = extract_metrics_from_group(group_data)
        
        if not metrics['experiment_num']:
            print(f"\n❌ No valid metrics in group {group_key}!")
            continue
        
        print(f"✓ Processing {len(metrics['experiment_num'])} experiments from group {group_key}\n")
        
        plot_comprehensive_analysis(
            metrics, 
            output_dir=args.output_dir,
            group_name=group_key
        )
        
        successful_graphs += 1
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Successfully generated {successful_graphs} graphs from {len(grouped_data)} groups")
    print(f"✓ Processed {len(args.log_path)} source director{'ies' if len(args.log_path) > 1 else 'y'}")
    
    if errors:
        print(f"\n⚠️  {len(errors)} files had parsing errors")
    
    print("\n" + "="*60)
    print(f"✓ All processing complete!")
    print("="*60)

if __name__ == "__main__":
    main()
