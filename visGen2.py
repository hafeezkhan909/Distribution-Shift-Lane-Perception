#!/usr/bin/env python3
import re, argparse, sys
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def slugify(txt): return re.sub(r'[^A-Za-z0-9._-]+','_',str(txt)).strip('_')

# ---------- Parsing helpers ----------
def parse_slurm_script(script_path: Path):
    """Return dict with log_name, dconfig, src, tgt; None if no log target."""
    c = script_path.read_text(errors='ignore')
    def g(pat):
        m = re.search(pat, c); return m.group(1) if m else None
    # log name: SBATCH --output/-o or shell redirect
    out = g(r"#SBATCH\s+(?:--output|-o)[=\s]+(\S+)")
    if not out:
        out = g(r">\s*([^\s]+\.log)")
    if not out:
        return None
    dconfig = g(r'--dConfig[=\s]+"?([A-Za-z0-9_]+)"?')
    src     = g(r'--src_samples[=\s]+"?(\d+)"?')
    tgt     = g(r'--tgt_samples[=\s]+"?(\d+)"?')
    return {
        "log_name": Path(out).name,
        "dconfig": dconfig,
        "src": int(src) if src else None,
        "tgt": int(tgt) if tgt else None,
        "__sh": str(script_path),
    }

def parse_log_metrics(log_path: Path):
    """Return dict with avg_mmd, std_mmd, tpr, tau (if found), else None."""
    txt = log_path.read_text(errors='ignore')
    m = {}
    tpr = re.search(r'TPR \(true positive rate\) over \d+ runs: ([\d.]+)%', txt)
    if tpr: m['tpr'] = float(tpr.group(1))
    testavg = re.search(r'Average MMD: ([\d.]+) ± ([\d.]+)', txt)
    if testavg:
        m['avg_mmd'] = float(testavg.group(1))
        m['std_mmd'] = float(testavg.group(2))
    tau = re.search(r'\[RESULT\] τ\([\d.]+\) = ([\d.]+)', txt)
    if tau: m['tau'] = float(tau.group(1))
    return m if m else None

# ---------- Load + enforce mapping ----------
def load_and_validate(log_roots, sh_roots, pattern='*.log', recursive=True):
    # 1. Collect all sh scripts
    sh_files = []
    for root in sh_roots:
        p = Path(root)
        if p.is_file() and p.suffix == '.sh':
            sh_files.append(p)
        elif p.is_dir():
            sh_files.extend(p.rglob('*.sh'))
    # 2. Parse sh -> truth, and build expected map
    truth = {}
    missing_fields = []
    for sh in sh_files:
        parsed = parse_slurm_script(sh)
        if not parsed:
            missing_fields.append(f"{sh}: no log target (#SBATCH --output or redirect)")
            continue
        if parsed['dconfig'] is None or parsed['src'] is None or parsed['tgt'] is None:
            missing_fields.append(f"{sh}: missing dconfig/src/tgt")
            continue
        truth[parsed['log_name']] = parsed

    # Collect all logs
    logs = []
    for root in log_roots:
        p = Path(root)
        if not p.exists(): continue
        logs.extend(list(p.rglob(pattern)) if recursive else [f for f in p.iterdir() if f.match(pattern)])

    # 3. Enforce mapping: every truth log must exist and parse metrics; no extras used
    errors = []
    groups = defaultdict(lambda: {"truth": None, "items": []})

    # Index logs by name for quick lookup
    log_index = {f.name: f for f in logs}

    # Check every truth entry has a log file
    for lname, tr in truth.items():
        if lname not in log_index:
            errors.append(f"Missing log file for {lname} (from {tr['__sh']})")
            continue
        lf = log_index[lname]
        metrics = parse_log_metrics(lf)
        if not metrics:
            errors.append(f"Metrics not found in {lname} (from {tr['__sh']})")
            continue
        key = f"{tr['dconfig']}_src{tr['src']}_tgt{tr['tgt']}"
        groups[key]["truth"] = tr
        groups[key]["items"].append({"log": lf, "metrics": metrics})

    # Any logs with no matching sh are ignored but warned
    for lname, lf in log_index.items():
        if lname not in truth:
            errors.append(f"Orphan log with no matching .sh: {lname}")

    # Accumulate missing field issues
    errors.extend(missing_fields)

    return groups, errors

# ---------- Plotting ----------
def plot_group(key, group, outdir):
    tr = group["truth"]
    items = group["items"]
    n = len(items)
    x = np.arange(n)
    tpr = [it["metrics"].get("tpr", 0) for it in items]
    avg = [it["metrics"].get("avg_mmd", 0) for it in items]
    tau = [it["metrics"].get("tau", 0) for it in items]
    seq_labels = [str(i+1) for i in range(n)]

    fig = plt.figure(figsize=(14,5))
    ax1 = plt.subplot(1,2,1)
    bars = ax1.bar(x, tpr, color='#D62828', edgecolor='black')
    for b, v in zip(bars, tpr):
        ax1.text(b.get_x()+b.get_width()/2., b.get_height()+1, f'{v:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_title('TPR (%)'); ax1.set_xticks(x); ax1.set_xticklabels(seq_labels); ax1.set_ylim(0, 110); ax1.axhline(100, ls='--', c='green', alpha=0.5)

    ax2 = plt.subplot(1,2,2)
    ax2.plot(x, avg, 'o-', color='#F18F01', label='Average MMD')
    if any(tau): ax2.plot(x, tau, 's--', color='#2E86AB', alpha=0.7, label='Tau')
    ax2.set_title('Average MMD'); ax2.set_xticks(x); ax2.set_xticklabels(seq_labels); ax2.legend()

    plt.suptitle(f"{tr['dconfig']} | Src={tr['src']} Tgt={tr['tgt']} | N={n}", fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])

    base = Path(outdir); base.mkdir(parents=True, exist_ok=True)
    fname_base = f"{slugify(key)}__n{n}"
    for fmt, dpi in [('png',300), ('svg',None), ('pdf',None), ('jpeg',300)]:
        out = base / f"{fname_base}.{fmt}"
        plt.savefig(out, dpi=dpi if dpi else None, bbox_inches='tight', format=fmt)
    plt.close()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-path', nargs='+', required=True)
    ap.add_argument('--sh-path', nargs='+', required=True)
    ap.add_argument('--recursive', action='store_true')
    ap.add_argument('--output-dir', default='figures_organized')
    ap.add_argument('--pattern', default='*.log')
    ap.add_argument('--expect-groups', type=int, default=None, help='Optional: abort if group count differs')
    args = ap.parse_args()

    groups, errors = load_and_validate(args.log_path, args.sh_path, args.pattern, args.recursive)
    total_groups = len(groups)

    print(f"\nDetected groups (key -> log count):")
    for k, g in groups.items():
        print(f"  {k} -> {len(g['items'])}")

    if errors:
        print(f"\n❌ Abort: {len(errors)} issues found. Fix these and rerun:")
        for e in errors[:50]:
            print("  " + e)
        if len(errors) > 50:
            print(f"  ... and {len(errors)-50} more")
        sys.exit(1)

    if args.expect_groups is not None and total_groups != args.expect_groups:
        print(f"\n❌ Abort: will produce {total_groups} groups, expected {args.expect_groups}.")
        sys.exit(1)

    # Proceed to plot
    for key, group in groups.items():
        if not group["items"]:
            print(f"Skipping empty group {key}")
            continue
        plot_group(key, group, args.output_dir)
    print(f"\n✓ Generated {total_groups} groups; PNGs in {args.output_dir}")

if __name__ == "__main__":
    main()
