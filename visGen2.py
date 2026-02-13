#!/usr/bin/env python3
"""
Visualize Mixed Shift Experiment Results (Robust Version with Slurm truth mapping)
- Discovers all Slurm .sh files, parses their ground-truth configs (dConfig, samples, num_runs, output log).
- Builds a mapping log_filename -> truth_from_sh.
- Reads .log files for MMD/TPR/tau and merges with truth (truth wins for configs; log wins for metrics).
- Warns on missing logs for a .sh, and on logs without a matching .sh.
"""

import re
import os
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Styles and configs
# ---------------------------------------------------------------------------
plt.rcParams['figure.figsize'] = (15, 10)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

ARCHITECTURE_CONFIGS = {
    "d128rel": {"name": "Remove Extra Layer", "layers": [4096, 1024, 128]},
    "d64rel": {"name": "Remove Extra Layer", "layers": [4096, 1024, 64]},
    "d32rel": {"name": "Remove Extra Layer", "layers": [4096, 1024, 32]},
    "d128gdd": {"name": "Gradually Decrease Dimensions", "layers": [4096, 512, 128]},
    "d64gdd": {"name": "Gradually Decrease Dimensions", "layers": [4096, 512, 64]},
    "d32gdd": {"name": "Gradually Decrease Dimensions", "layers": [4096, 512, 32]},
    "d64ids": {"name": "Increase Dimension Size", "layers": [4096, 1024, 256, 64]},
    "d128ids": {"name": "Increase Dimension Size", "layers": [4096, 1024, 256, 128]},
    "d32ids": {"name": "Increase Dimension Size", "layers": [4096, 1024, 256, 32]},
    "d32": {"name": "32D Standard", "layers": [4096, 1024, 256, 32]},
    "orig": {"name": "Original Architecture", "layers": [4096, 1024, 256]},
}

# Slurm argument patterns
SLURM_ARG_PATTERNS = {
    'dconfig': r'--dConfig\s+([A-Za-z0-9_]+)',
    'src_samples': r'--src_samples\s+(\d+)',
    'tgt_samples': r'--tgt_samples\s+(\d+)',
    'ratio_src_samples': r'--ratio_src_samples\s+(\d+)',
    'ratio_tgt_samples': r'--ratio_tgt_samples\s+(\d+)',
    'num_runs': r'--num_runs\s+(\d+)',
    'file_name': r'--file_name\s+"?([\w\.-]+)"?',
    'output': r'#SBATCH\s+--output\s+([^\s]+)',
}
OUTPUT_REDIRECT_PATTERN = r'>\s*([^\s]+\.log)'

# ---------------------------------------------------------------------------
# Helpers: Slurm parsing
# ---------------------------------------------------------------------------
def parse_slurm_script(script_path: Path):
    """Parse a Slurm bash script for ground-truth args and output log name."""
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return {}

    out = {}
    for key, pat in SLURM_ARG_PATTERNS.items():
        m = re.search(pat, content)
        if m:
            val = m.group(1)
            out[key] = int(val) if val.isdigit() else val

    # Try to catch shell redirect like: python ... > something.log
    redir = re.search(OUTPUT_REDIRECT_PATTERN, content)
    if redir and 'output' not in out:
        out['output'] = redir.group(1)
    return out

def discover_slurm_truth(sh_paths):
    """
    Discover all .sh files, build mapping log_filename -> truth dict.
    Returns (truth_map, warnings)
    """
    truth = {}
    warnings = []
    sh_files = []
    for p in sh_paths:
        pth = Path(p)
        if pth.is_file() and pth.suffix == '.sh':
            sh_files.append(pth)
        elif pth.is_dir():
            sh_files.extend(pth.rglob('*.sh'))
    print(f"Discovered {len(sh_files)} slurm scripts")

    for sh in sh_files:
        parsed = parse_slurm_script(sh)
        if not parsed:
            warnings.append(f"{sh}: could not parse Slurm script")
            continue
        log_name = parsed.get('output') or parsed.get('file_name')
        if not log_name:
            warnings.append(f"{sh}: no #SBATCH --output or file_name found; cannot map to log")
            continue
        log_basename = Path(log_name).name
        parsed['__source_sh'] = str(sh)
        truth[log_basename] = parsed
    return truth, warnings

# ---------------------------------------------------------------------------
# Helpers: filename-based config extraction (fallback)
# ---------------------------------------------------------------------------
def extract_config_from_filename(filename):
    stem = Path(filename).stem
    config = {
        'dim_config': 'unknown',
        'arch_type': 'unknown',
        'calibration_samples': 'unknown',
        'experiment_num': None
    }
    dim_match = re.search(r'd(\d+)(ids|gdd|rel)', stem, re.IGNORECASE)
    if dim_match:
        dim_size = dim_match.group(1)
        dim_type = dim_match.group(2).lower()
        config['dim_config'] = f"d{dim_size}{dim_type}"
        config['arch_type'] = dim_type
    k_match = re.search(r'K(\d+)', stem)
    if k_match:
        config['calibration_samples'] = int(k_match.group(1))
    else:
        digit_match = re.search(r'd\d+(?:ids|gdd|rel)(\d+)', stem)
        if digit_match:
            config['calibration_samples'] = int(digit_match.group(1))
    exp_nums = re.findall(r'(\d+)', stem)
    if exp_nums:
        for num in exp_nums:
            num_int = int(num)
            if num_int not in [10, 32, 64, 100, 128, 1000]:
                config['experiment_num'] = num_int
                break
        if config['experiment_num'] is None and exp_nums:
            config['experiment_num'] = int(exp_nums[0])
    return config

# ---------------------------------------------------------------------------
# Helpers: log parsing
# ---------------------------------------------------------------------------
def parse_log_file(log_path):
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(log_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
    metrics = {}
    exp_header = re.search(r'Experiment\s+(\d+):\s+Src=[\d.]+%\s+\((\d+)\s+samples?\),\s+Tgt=[\d.]+%\s+\((\d+)\s+samples?\)', content)
    if exp_header:
        metrics['experiment_num'] = int(exp_header.group(1))
        metrics['src_samples'] = int(exp_header.group(2))
        metrics['tgt_samples'] = int(exp_header.group(3))
    ae_dim_match = re.search(r'features loaded\. Shape = \(\d+, (\d+)\)', content)
    if ae_dim_match:
        metrics['autoencoder_dim'] = int(ae_dim_match.group(1))
    intended_dim_match = re.search(r'\[ResNet-AE\] dimensionallity: (\d+)', content)
    if intended_dim_match:
        metrics['intended_dim'] = int(intended_dim_match.group(1))
    src_calib_match = re.search(r'CULane features loaded\. Shape = \((\d+), \d+\)', content)
    if src_calib_match:
        metrics['src_calib'] = int(src_calib_match.group(1))
    tgt_calib_match = re.search(r'Total combined samples: (\d+)', content)
    if tgt_calib_match:
        metrics['tgt_calib'] = int(tgt_calib_match.group(1))
    tau_match = re.search(r'\[RESULT\] τ\([\d.]+\) = ([\d.]+)', content)
    if tau_match:
        metrics['tau'] = float(tau_match.group(1))
    calib_match = re.search(r'Mean MMD \(same-distribution\): ([\d.]+) ± ([\d.]+)', content)
    if calib_match:
        metrics['calib_mmd'] = float(calib_match.group(1))
        metrics['calib_std'] = float(calib_match.group(2))
    sanity_match = re.search(r'\[SANITY CHECK\] MMD.*? = ([\d.]+)', content)
    if sanity_match:
        metrics['sanity_mmd'] = float(sanity_match.group(1))
    test_avg_match = re.search(r'Average MMD: ([\d.]+) ± ([\d.]+)', content)
    if test_avg_match:
        metrics['test_avg_mmd'] = float(test_avg_match.group(1))
        metrics['test_std_mmd'] = float(test_avg_match.group(2))
    tpr_match = re.search(r'TPR \(true positive rate\) over \d+ runs: ([\d.]+)%', content)
    if tpr_match:
        metrics['tpr'] = float(tpr_match.group(1))
    return metrics if metrics else None

# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------
def get_config_description(arch_type, ae_dim):
    if not arch_type or arch_type == 'unknown':
        return f"{ae_dim}D - Original" if ae_dim != 'unknown' else "Unknown"
    type_mapping = {"rel": "Remove Extra Layer", "ids": "Increase Dimension Size", "gdd": "Gradually Decrease Dimensions"}
    type_name = type_mapping.get(arch_type, arch_type.upper())
    return f"{ae_dim}D - {type_name}"

def get_architecture_text(arch_type, ae_dim):
    config_key = f"d{ae_dim}{arch_type}" if arch_type != 'unknown' else f"d{ae_dim}"
    if config_key in ARCHITECTURE_CONFIGS:
        config = ARCHITECTURE_CONFIGS[config_key]
        name = config['name']
        layers = config['layers']
        layer_text = " → ".join(str(layer) for layer in layers)
        return f"{name}\n{layer_text}"
    type_mapping = {"rel": "Remove Extra Layer", "ids": "Increase Dimension Size", "gdd": "Gradually Decrease Dimensions"}
    type_name = type_mapping.get(arch_type, "Unknown Architecture")
    return f"{type_name}\n(Configuration: {ae_dim}D)"

def create_config_key(file_config, log_metrics):
    arch_type = file_config.get('arch_type', log_metrics.get('arch_type', 'unknown'))
    ae_dim = log_metrics.get('autoencoder_dim', 'unknown')
    if 'dim_config' in file_config and file_config['dim_config'] != 'unknown':
        dim_key = file_config['dim_config']
        dim_num = re.search(r'd(\d+)', dim_key, re.IGNORECASE)
        if dim_num:
            ae_dim = int(dim_num.group(1))
    src_calib = log_metrics.get('src_calib', file_config.get('calibration_samples', file_config.get('src_samples_truth', 'unknown')))
    tgt_calib = log_metrics.get('tgt_calib', file_config.get('calibration_samples', file_config.get('tgt_samples_truth', 'unknown')))
    config_descriptor = f"{arch_type}{ae_dim}d" if arch_type != 'unknown' else f"{ae_dim}d"
    return f"{config_descriptor}__ae{ae_dim}_src{src_calib}_tgt{tgt_calib}"

# ---------------------------------------------------------------------------
# Load and group data
# ---------------------------------------------------------------------------
def load_experiment_data_grouped(logs_dirs, sh_dirs, log_pattern='*.log', recursive=True):
    errors = []
    warnings = []

    # Discover Slurm truth
    slurm_truth, slurm_warnings = discover_slurm_truth(sh_dirs)
    warnings.extend(slurm_warnings)

    # Collect log files
    all_log_files = []
    for logs_dir in logs_dirs:
        logs_path = Path(logs_dir)
        if not logs_path.exists():
            errors.append(f"Directory not found: {logs_dir}")
            print(f"⚠ Directory not found: {logs_dir}")
            continue
        if recursive:
            log_files = list(logs_path.rglob(log_pattern))
        else:
            log_files = [f for f in logs_path.iterdir() if f.is_file() and f.match(log_pattern)]
        for log_file in log_files:
            all_log_files.append((log_file, logs_dir))
        print(f"Found {len(log_files)} log files in {logs_dir}")
    all_log_files.sort(key=lambda x: x[0].name)
    total_files = len(all_log_files)
    print(f"\nTotal log files to process: {total_files}")

    grouped_data = defaultdict(lambda: {'experiments': {}, 'metadata': {}, 'source_dirs': set()})
    seen_logs = set()

    for log_file, source_dir in all_log_files:
        seen_logs.add(log_file.name)
        # Start with filename-based config
        file_config = extract_config_from_filename(log_file.name)

        # If Slurm truth is available for this log filename, override configs
        if log_file.name in slurm_truth:
            truth = slurm_truth[log_file.name]
            if 'dconfig' in truth:
                file_config['dim_config'] = truth['dconfig']
                m_arch = re.search(r'(ids|gdd|rel)', truth['dconfig'], re.IGNORECASE)
                if m_arch:
                    file_config['arch_type'] = m_arch.group(1).lower()
            if 'src_samples' in truth:
                file_config['src_samples_truth'] = truth['src_samples']
            if 'tgt_samples' in truth:
                file_config['tgt_samples_truth'] = truth['tgt_samples']
            if 'ratio_src_samples' in truth:
                file_config['ratio_src_truth'] = truth['ratio_src_samples']
            if 'ratio_tgt_samples' in truth:
                file_config['ratio_tgt_truth'] = truth['ratio_tgt_samples']
            if 'num_runs' in truth:
                file_config['num_runs_truth'] = truth['num_runs']
            file_config['__source_sh'] = truth.get('__source_sh')

        display_path = log_file.name
        print(f"[{log_file.name}] ", end='', flush=True)

        try:
            metrics = parse_log_file(log_file)
            if file_config['arch_type'] != 'unknown':
                print(f"[Type: {file_config['arch_type']}] ", end='', flush=True)
            if metrics:
                src = metrics.get('src_samples', None)
                tgt = metrics.get('tgt_samples', None)
                mmd = metrics.get('test_avg_mmd', None)
                tpr = metrics.get('tpr', None)
                ae_dim = metrics.get('autoencoder_dim', None)
                intended_dim = metrics.get('intended_dim', None)
                src_calib = metrics.get('src_calib', None)
                tgt_calib = metrics.get('tgt_calib', None)

                if intended_dim is not None and ae_dim is not None and intended_dim != ae_dim:
                    warnings.append(f"{display_path}: Dimension mismatch! Intended={intended_dim}D, Loaded={ae_dim}D")
                    print(f"⚠️ [DIM MISMATCH intended {intended_dim}D, loaded {ae_dim}D] ", end='', flush=True)

                # Truth overrides from Slurm
                if 'src_samples_truth' in file_config:
                    if src is not None and file_config['src_samples_truth'] != src:
                        warnings.append(f"{display_path}: src_samples mismatch log={src} sh={file_config['src_samples_truth']}")
                    metrics['src_samples'] = file_config['src_samples_truth']
                if 'tgt_samples_truth' in file_config:
                    if tgt is not None and file_config['tgt_samples_truth'] != tgt:
                        warnings.append(f"{display_path}: tgt_samples mismatch log={tgt} sh={file_config['tgt_samples_truth']}")
                    metrics['tgt_samples'] = file_config['tgt_samples_truth']
                if 'num_runs_truth' in file_config:
                    metrics['num_runs'] = file_config['num_runs_truth']
                if 'dim_config' in file_config and file_config['dim_config'] != 'unknown':
                    metrics['dim_config'] = file_config['dim_config']
                    metrics['arch_type'] = file_config.get('arch_type', metrics.get('arch_type', 'unknown'))

                # Refresh src/tgt after overrides
                src = metrics.get('src_samples', None)
                tgt = metrics.get('tgt_samples', None)

                exp_num = metrics.get('experiment_num', file_config.get('experiment_num', 0))
                if src is not None and tgt is not None and mmd is not None:
                    metrics['file_config'] = file_config
                    metrics['arch_type'] = file_config.get('arch_type', metrics.get('arch_type', 'unknown'))
                    config_key = create_config_key(file_config, metrics)
                    unique_key = log_file.stem
                    grouped_data[config_key]['experiments'][unique_key] = {
                        'metrics': metrics,
                        'file_path': log_file,
                        'exp_num': exp_num,
                        'source_dir': source_dir
                    }
                    grouped_data[config_key]['source_dirs'].add(str(Path(source_dir)))
                    if not grouped_data[config_key]['metadata']:
                        grouped_data[config_key]['metadata'] = {
                            'ae_dim': ae_dim,
                            'src_calib': src_calib,
                            'tgt_calib': tgt_calib,
                            'arch_type': metrics.get('arch_type', file_config.get('arch_type', 'unknown'))
                        }
                    print(f"✓ Group: {config_key} (Exp={exp_num}, Src={src}, Tgt={tgt}, TPR={tpr}%)")
                else:
                    errors.append(f"{display_path}: Missing required data (src={src}, tgt={tgt}, mmd={mmd})")
                    print(f"⚠ MISSING: src={src}, tgt={tgt}, mmd={mmd}")
            else:
                errors.append(f"{display_path}: Failed to parse")
                print(f"✗ Failed to parse")
        except Exception as e:
            errors.append(f"{display_path}: Exception - {str(e)}")
            print(f"✗ Exception: {str(e)}")

    # Warn about slurm scripts whose logs were not found
    for log_basename, truth in slurm_truth.items():
        if log_basename not in seen_logs:
            warnings.append(f"Slurm {truth.get('__source_sh','(unknown)')} points to missing log: {log_basename}")

    # Finalize grouped data
    result = dict(grouped_data)
    for config_key in result:
        result[config_key]['source_dirs'] = list(result[config_key]['source_dirs'])

    print("\n" + "="*60)
    print(f"GROUPING SUMMARY (from {len(logs_dirs)} log roots, {total_files} log files)")
    print("="*60)
    for config_key, group_data in result.items():
        n_experiments = len(group_data['experiments'])
        metadata = group_data['metadata']
        source_dirs = group_data['source_dirs']
        print(f"\nGroup: {config_key}")
        print(f"  Architecture Type: {metadata.get('arch_type', 'unknown')}")
        print(f"  AE Dimension (actual): {metadata.get('ae_dim', 'unknown')}")
        print(f"  Source Calibration: {metadata.get('src_calib', 'unknown')}")
        print(f"  Target Calibration: {metadata.get('tgt_calib', 'unknown')}")
        print(f"  Number of experiments: {n_experiments}")
        print(f"  Source directories: {len(source_dirs)}")
        for src_dir in source_dirs:
            print(f"    - {src_dir}")

    if warnings:
        print("\n" + "="*60)
        print(f"⚠️  WARNINGS ({len(warnings)} detected):")
        print("="*60)
        for w in warnings[:50]:
            print(f"  {w}")
        if len(warnings) > 50:
            print(f"  ... and {len(warnings) - 50} more")

    return result, errors

# ---------------------------------------------------------------------------
# Metrics extraction and plotting
# ---------------------------------------------------------------------------
def extract_metrics_from_group(group_data):
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
        'arch_type': [],
        'file_paths': [],
        'source_dirs': []
    }
    for unique_key, exp_data in sorted(experiments.items(), key=lambda x: x[1]['exp_num']):
        exp_metrics = exp_data['metrics']
        src = exp_metrics.get('src_samples', 0)
        tgt = exp_metrics.get('tgt_samples', 0)
        if 'test_avg_mmd' not in exp_metrics:
            continue
        ratio = tgt / src if src > 0 else (float('inf') if tgt > 0 else 0)
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
        arch_type = exp_metrics.get('arch_type', 'unknown')
        metrics['arch_type'].append(arch_type)
        metrics['file_paths'].append(str(exp_data['file_path']))
        metrics['source_dirs'].append(exp_data.get('source_dir', 'unknown'))
    return metrics

def create_output_directories(base_dir='figures'):
    base_path = Path(base_dir)
    for fmt in ['svg', 'png', 'jpeg', 'pdf', 'eps']:
        (base_path / fmt).mkdir(parents=True, exist_ok=True)
    return base_path

def generate_filename(ae_dim, src_calib, tgt_calib, n_experiments, arch_type=None):
    if arch_type and arch_type != 'unknown':
        base = f"mmd_analysis_{arch_type}_dim{ae_dim}_src{src_calib}_tgt{tgt_calib}_n{n_experiments}"
    else:
        base = f"mmd_analysis_dim{ae_dim}_src{src_calib}_tgt{tgt_calib}_n{n_experiments}"
    return base

def plot_comprehensive_analysis(metrics, output_dir='figures', group_name='experiment'):
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
    sorted_arch_type = [metrics['arch_type'][i] for i in sorted_indices]
    sorted_file_paths = [metrics['file_paths'][i] for i in sorted_indices]
    sorted_source_dirs = [metrics['source_dirs'][i] for i in sorted_indices]

    ae_dim_mode = sorted_ae_dim[0] if sorted_ae_dim else 0
    src_calib_mode = sorted_src_calib[0] if sorted_src_calib else 0
    tgt_calib_mode = sorted_tgt_calib[0] if sorted_tgt_calib else 0
    arch_type_mode = sorted_arch_type[0] if sorted_arch_type else 'unknown'
    sequential_labels = [f"{i+1}" for i in range(n_experiments)]

    fig = plt.figure(figsize=(18, 8))

    # TPR
    ax1 = plt.subplot(1, 3, 1)
    colors = ['#06A77D' if tpr == 100 else '#D62828' for tpr in sorted_tpr]
    bars = plt.bar(x_positions, sorted_tpr, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    for bar, tpr in zip(bars, sorted_tpr):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{tpr:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.xlabel('Experiment Number (Ordered by Ratio)', fontsize=13, fontweight='bold')
    plt.ylabel('TPR (%)', fontsize=14, fontweight='bold')
    plt.title('True Positive Rate', fontsize=16, fontweight='bold', pad=15)
    plt.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Perfect Detection')
    plt.xticks(x_positions, sequential_labels, fontsize=10)
    plt.ylim([0, 110])
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=11)

    # MMD
    ax2 = plt.subplot(1, 3, 2)
    plt.plot(x_positions, sorted_avg_mmd, 'o-', linewidth=2.5, markersize=10, color='#F18F01', label='Average MMD')
    plt.plot(x_positions, sorted_tau, 's--', linewidth=2, markersize=8, color='#2E86AB', alpha=0.7, label='Tau (Threshold)')
    plt.xlabel('Experiment Number (Ordered by Ratio)', fontsize=13, fontweight='bold')
    plt.ylabel('MMD Value', fontsize=14, fontweight='bold')
    plt.title('Average MMD vs Sample Ratio', fontsize=16, fontweight='bold', pad=15)
    plt.xticks(x_positions, sequential_labels, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='best')

    # Samples
    ax3 = plt.subplot(1, 3, 3)
    width = 0.25
    bars1 = plt.bar(x_positions - width/2, sorted_src, width, label='Source Samples', alpha=0.8, color='#06A77D', edgecolor='black')
    bars2 = plt.bar(x_positions + width/2, sorted_tgt, width, label='Target Samples', alpha=0.8, color='#F18F01', edgecolor='black')
    max_height = max(max(sorted_src), max(sorted_tgt)) if sorted_src and sorted_tgt else 1
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        plt.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + max_height * 0.015, f'{int(sorted_src[i])}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#06A77D')
        plt.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + max_height * 0.015, f'{int(sorted_tgt[i])}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#F18F01')
    for i in range(len(sorted_exp_num)):
        ratio_text = '∞' if sorted_ratio[i] == float('inf') else f'{sorted_ratio[i]:.2f}'
        plt.text(x_positions[i], -max_height * 0.08, ratio_text, ha='center', va='top', fontsize=8, style='italic', color='#555')
    plt.xlabel('Experiment Number (Ordered by Ratio)\n(Ratio shown below)', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=14, fontweight='bold')
    plt.title('Source vs Target Samples', fontsize=16, fontweight='bold', pad=15)
    plt.xticks(x_positions, sequential_labels, fontsize=10)
    plt.xlim([-0.7, n_experiments - 0.3])
    plt.ylim([-max_height * 0.12, max_height * 1.08])
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, axis='y')

    config_desc = get_config_description(arch_type_mode, ae_dim_mode)
    unique_sources = set(sorted_source_dirs)
    n_sources = len(unique_sources)
    source_note = f" (from {n_sources} source dir{'s' if n_sources > 1 else ''})" if n_sources > 1 else ""
    plt.suptitle(
        f'Maximum Mean Discrepancy Analysis of Distribution Shift Detection\n'
        f'Architecture: {config_desc} | '
        f'Source Samples: {src_calib_mode} | '
        f'Target Samples: {tgt_calib_mode} | '
        f'N={n_experiments} Experiments{source_note}',
        fontsize=16, fontweight='bold', y=0.97
    )

    arch_text = get_architecture_text(arch_type_mode, ae_dim_mode)
    fig.text(0.5, 0.02, arch_text,
             ha='center', va='bottom',
             fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    base_filename = generate_filename(ae_dim_mode, src_calib_mode, tgt_calib_mode, n_experiments, arch_type_mode)
    formats = {'svg': {'dpi': None}, 'png': {'dpi': 300}, 'jpeg': {'dpi': 300}, 'pdf': {'dpi': None}, 'eps': {'dpi': None}}
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
        ratio_str = '∞' if sorted_ratio[i] == float('inf') else f'{sorted_ratio[i]:.2f}'
        arch_str = sorted_arch_type[i] if sorted_arch_type[i] != 'unknown' else 'N/A'
        file_path = Path(sorted_file_paths[i])
        source_dir = Path(sorted_source_dirs[i]).name if sorted_source_dirs[i] != 'unknown' else 'N/A'
        print(f"  {i+1:2d} → Exp{actual_exp:2d} [{file_path.name}] from [{source_dir}] (Type: {arch_str}, Dim: {sorted_ae_dim[i]}, Src: {sorted_src[i]}, Tgt: {sorted_tgt[i]}, Ratio: {ratio_str})")

    plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Visualize experiment results using Slurm truth + log metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python visualize_shift_results.py \\
    --log-path /home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash/IncreaseDimentionallity \\
               /home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash/GraduallyDecreaseDimensions \\
               /home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash/RemoveExtraLayer \\
    --sh-path  /home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash \\
    --recursive \\
    --output-dir figures_organized
        """
    )
    parser.add_argument('--log-path', type=str, nargs='+', required=True, help='Path(s) to directories containing log files')
    parser.add_argument('--sh-path', type=str, nargs='+', help='Path(s) to search for Slurm .sh files (defaults to log-path)', default=None)
    parser.add_argument('--recursive', action='store_true', help='Search recursively for log files')
    parser.add_argument('--output-dir', type=str, default='figures', help='Directory to save output figures')
    parser.add_argument('--pattern', type=str, default='*.log', help='Glob pattern for log files (default: *.log)')
    args = parser.parse_args()

    sh_paths = args.sh_path if args.sh_path else args.log_path

    print("\n" + "="*60)
    print(f"Processing log files from {len(args.log_path)} root(s)")
    for p in args.log_path:
        print(f"  - {p}")
    print(f"Slurm search roots: {', '.join(sh_paths)}")
    print(f"Recursive: {args.recursive}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)

    grouped_data, errors = load_experiment_data_grouped(args.log_path, sh_paths, args.pattern, args.recursive)

    if errors:
        print(f"\n⚠️  Encountered {len(errors)} errors during parsing:")
        for e in errors[:20]:
            print(f"   - {e}")
        if len(errors) > 20:
            print(f"   ... and {len(errors) - 20} more")

    if not grouped_data:
        print("\n❌ No valid experiment groups found!")
        return

    print("\n" + "="*60)
    print(f"Found {len(grouped_data)} distinct experiment configurations")
    print("="*60)

    successful_graphs = 0
    for group_key, group_data in grouped_data.items():
        print("\n" + "="*60)
        print(f"GENERATING GRAPH FOR: {group_key}")
        print("="*60)
        metrics = extract_metrics_from_group(group_data)
        if not metrics['experiment_num']:
            print(f"❌ No valid metrics in group {group_key}!")
            continue
        print(f"✓ Processing {len(metrics['experiment_num'])} experiments from group {group_key}\n")
        plot_comprehensive_analysis(metrics, output_dir=args.output_dir, group_name=group_key)
        successful_graphs += 1

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Successfully generated {successful_graphs} graphs from {len(grouped_data)} groups")
    print(f"✓ Processed {len(args.log_path)} log root(s)")

    if errors:
        print(f"\n⚠️  {len(errors)} files had parsing errors")

    print("\n" + "="*60)
    print("✓ All processing complete!")
    print("="*60)

if __name__ == "__main__":
    main()
