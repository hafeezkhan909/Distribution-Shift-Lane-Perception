import os
import stat
import subprocess
import sys

# --- CONFIGURATION ---
# The location of your FLAT logs and STRUCTURED script folders
BASE_DIR = "/home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash"
BATCH_SIZE = 512  # Safe size to prevent OOM
# ---------------------

configs = [
    "d128rel", "d64rel", "d32rel", 
    "d128gdd", "d64gdd", "d32gdd", 
    "d64ids", "d128ids", "d32"
]
sample_sizes = [10, 100, 1000]

submitted_count = 0

print(f"--- Scanning {BASE_DIR} for missing jobs ---")

for config in configs:
    # 1. Determine Subfolder Name
    if "ids" in config: type_dir = "IncreaseDimensionality"
    elif "rel" in config: type_dir = "RemoveExtraLayer"
    elif "gdd" in config: type_dir = "GraduallyDecreaseDimensions"
    elif "d32" == config: type_dir = "Base32"
    else: type_dir = "Misc"

    for K in sample_sizes:
        # 2. Define Paths
        script_dir = os.path.join(BASE_DIR, type_dir, f"{config}_K{K}")
        srcSamples = K
        
        # Check all 11 experiments (Exp1 to Exp11)
        exp_count = 1
        for src_percent in range(0, 101, 10):
            tgt_percent = 100 - src_percent
            ratio_src = int(K * src_percent / 100)
            ratio_tgt = int(K * tgt_percent / 100)
            
            job_name = f"{config}_K{K}_Exp{exp_count}"
            log_filename = f"{job_name}.log"
            script_filename = f"{job_name}.sh"
            
            # ABSOLUTE PATHS (Crucial for correct Slurm behavior)
            log_path = os.path.join(BASE_DIR, log_filename)         # Flat Log
            script_path = os.path.join(script_dir, script_filename) # Structured Script
            
            # --- DIAGNOSTICS ---
            needs_run = False
            
            # Check A: Does the Log exist in LocalBash root?
            if not os.path.exists(log_path):
                needs_run = True
            else:
                # Check B: Did it finish successfully?
                try:
                    with open(log_path, 'r', errors='ignore') as f:
                        content = f.read()
                        if "Job finished" not in content:
                            needs_run = True # Crashed or Incomplete
                        if "CUDA out of memory" in content:
                            needs_run = True # Definitely Rerun
                except:
                    needs_run = True

            # --- RECOVERY ---
            if needs_run:
                # Ensure directory exists
                os.makedirs(script_dir, exist_ok=True)
                
                # Regenerate Script
                with open(script_path, 'w') as f:
                    f.write(f"""#!/bin/bash
#SBATCH --job-name={config}_K{K}_E{exp_count}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output={log_path}
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

echo "----------------------------------------------------"
echo "Recovering Job: {config} | K={K} | Exp={exp_count}"
echo "----------------------------------------------------"

export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh
conda activate ml_project
cd /home1/adoyle2025/Distribution-Shift-Lane-Perception

python shift_concat_experiment.py \\
    --source_dir /home1/adoyle2025/Datasets/Datasets/CULane \\
    --target_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes \\
    --source_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt \\
    --target_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \\
    --source_test_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/test.txt \\
    --src_samples {srcSamples} \\
    --tgt_samples {K} \\
    --ratio_src_samples {ratio_src} \\
    --ratio_tgt_samples {ratio_tgt} \\
    --num_runs 100 \\
    --block_idx 4 \\
    --seed_base 32 \\
    --batch_size {BATCH_SIZE} \\
    --dConfig "{config}" \\
    --save_all_image_paths True \\
    --file_name "{config}_K{K}.json"

echo "Job finished: $(date)"
""")
                # Make executable
                st = os.stat(script_path)
                os.chmod(script_path, st.st_mode | stat.S_IEXEC)
                
                # --- AUTO SUBMIT ---
                try:
                    # Execute sbatch directly
                    subprocess.run(["sbatch", script_path], check=True, stdout=subprocess.DEVNULL)
                    
                    # Print status (overwrite line for clean output)
                    sys.stdout.write(f"\rSubmitted: {job_name}      ")
                    sys.stdout.flush()
                    submitted_count += 1
                except subprocess.CalledProcessError as e:
                    print(f"\nError submitting {job_name}: {e}")

            exp_count += 1

print(f"\n------------------------------------------------")
print(f"Recovery Complete.")
print(f"Total jobs submitted: {submitted_count}")