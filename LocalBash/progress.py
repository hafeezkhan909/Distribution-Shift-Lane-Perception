#!/usr/bin/env python3
"""
Experiment Monitor Dashboard (Flat Logs Version)
- Scans subfolders to find PLANNED experiments.
- Scans the root LocalBash folder to find the ACTUAL logs.
- Visualizes progress of all configs.
"""

import os
import time
import sys
from pathlib import Path
from datetime import datetime

# --- Config ---
BASE_PATH = Path("./LocalBash")  # Root folder
REFRESH_RATE = 10  # Seconds between updates
TOTAL_EXPS_PER_BATCH = 11  # 0% to 100% in 10% steps

# --- Colors ---
class C:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    CLR = '\033[2J\033[H' # Clear screen

def get_status(log_path):
    """
    Returns: 'Done', 'Error', 'Running', 'Pending'
    """
    if not log_path.exists():
        return "Pending"
    
    try:
        with open(log_path, 'r', errors='ignore') as f:
            content = f.read()
            
        if "Job finished:" in content:
            return "Done"
        # Check for python errors or slurm timeouts
        elif "Traceback" in content or "CANCELLED" in content or "error" in content.lower():
            # Filter out benign "error" words if necessary
            if "standard error" in content.lower(): return "Running" # Slurm header
            return "Error"
        else:
            return "Running"
    except:
        return "Unknown"

def draw_progress_bar(count, total, width=20):
    if total == 0: total = 1 # Avoid div by zero
    filled = int(round(width * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '█' * filled + '-' * (width - filled)
    return f"[{bar}] {percents}%"

def scan_system():
    groups = []
    
    # 1. Discover Groups by looking for folders with .sh files
    # We walk the directory tree to find the experiment folders (e.g., d128ids_K10)
    for root, dirs, files in os.walk(BASE_PATH):
        if any(f.endswith('.sh') for f in files):
            path = Path(root)
            folder_name = path.name # e.g., "d128ids_K10"
            
            # Skip the root LocalBash folder itself if it contains stray sh files
            if path == BASE_PATH:
                continue

            stats = {
                "name": folder_name,
                "pending": 0,
                "running": 0,
                "done": 0,
                "error": 0,
                "total": TOTAL_EXPS_PER_BATCH
            }
            
            # 2. Check for Logs in the ROOT (LocalBash)
            # The naming convention matches the folder name: {FolderName}_Exp{i}.log
            # Example: Folder "d128ids_K10" -> Log "d128ids_K10_Exp1.log"
            
            for i in range(1, TOTAL_EXPS_PER_BATCH + 1):
                log_name = f"{folder_name}_Exp{i}.log"
                
                # IMPORTANT: Look in BASE_PATH, not the subfolder
                log_path = BASE_PATH / log_name
                
                status = get_status(log_path)
                
                if status == "Done": stats["done"] += 1
                elif status == "Error": stats["error"] += 1
                elif status == "Running": stats["running"] += 1
                else: stats["pending"] += 1
            
            groups.append(stats)
            
    # Sort by Name for consistent display
    groups.sort(key=lambda x: x['name'])
    return groups

def print_dashboard(groups):
    print(C.CLR)
    print(f"{C.HEADER}{C.BOLD}Distribution Shift Monitor (Logs in Root){C.ENDC}")
    print(f"Location: {BASE_PATH.resolve()}")
    print(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 90)
    
    print(f"{'Configuration':<35} | {'Progress':<30} | {'Status (D/R/P/E)':<18}")
    print("-" * 90)
    
    total_done = 0
    total_jobs = 0
    
    for g in groups:
        name = g['name']
        d, r, p, e = g['done'], g['running'], g['pending'], g['error']
        total = g['total']
        
        # Color Logic
        status_color = C.OKGREEN
        if e > 0: status_color = C.FAIL      # RED if any error
        elif r > 0: status_color = C.WARNING # YELLOW if running
        elif d == total: status_color = C.OKBLUE # BLUE if all done
        elif d == 0 and r == 0: status_color = C.ENDC # GREY if nothing started
        
        p_bar = draw_progress_bar(d, total)
        
        # Status Numbers
        status_str = f"{d}/{r}/{p}/{e}"
        if e > 0: status_str = f"{C.FAIL}{status_str}{C.ENDC}"
        elif r > 0: status_str = f"{C.WARNING}{status_str}{C.ENDC}"
        elif d == total: status_str = f"{C.OKBLUE}{status_str}{C.ENDC}"
        
        print(f"{status_color}{name:<35}{C.ENDC} | {p_bar:<30} | {status_str:<18}")
        
        total_done += d
        total_jobs += total

    print("-" * 90)
    if total_jobs > 0:
        global_progress = draw_progress_bar(total_done, total_jobs, width=50)
        print(f"{C.BOLD}Global Progress: {global_progress}{C.ENDC}")
    else:
        print("No experiments found. Check if .sh files exist in subfolders.")
        
    print(f"\n{C.OKBLUE}D=Done{C.ENDC}, {C.WARNING}R=Running{C.ENDC}, P=Pending, {C.FAIL}E=Error{C.ENDC}")
    print("Press Ctrl+C to exit.")

def main():
    try:
        if not BASE_PATH.exists():
            print(f"Error: Directory {BASE_PATH} does not exist.")
            return

        while True:
            groups = scan_system()
            if not groups:
                print(f"Scanning {BASE_PATH}... found no experiment subfolders yet.")
                time.sleep(2)
                continue
            
            print_dashboard(groups)
            time.sleep(REFRESH_RATE)
    except KeyboardInterrupt:
        print("\nExiting Monitor.")
        sys.exit(0)

if __name__ == "__main__":
    main()