import os

# --- Configuration & Paths ---
BASE_ROOT = "/home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash/ModelExperiments/Phase 1"
PROJECT_DIR = "/home1/adoyle2025/Distribution-Shift-Lane-Perception"
CONDA_PROFILE = "/home1/adoyle2025/miniconda3/etc/profile.d/conda.sh"

MODELS = {
    "CULaneModel": "CU_Lane",
    "CurvelanesModel": "CurveLanes",
    "ImageNetModel": "ImageNet",
    "RandomWeightsModel": "Random",
    "AssistTaxiModel": "ASSIST_Taxi"
}

DATA_MAP = {
    "CULanes": (
        "/home1/adoyle2025/Datasets/Datasets/CULane",
        "/home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt",
        "/home1/adoyle2025/Datasets/Datasets/CULane/list/test.txt"
    ),
    "Curvelanes": (
        "/home1/adoyle2025/Datasets/Datasets/Curvelanes",
        "/home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt",
        "/home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt"
    ),
    "AssistTaxi": (
        "/home1/adoyle2025/Datasets/Datasets/ASSIST-Taxi",
        "/home1/adoyle2025/Datasets/Datasets/ASSIST-Taxi/train.txt",
        "/home1/adoyle2025/Datasets/Datasets/ASSIST-Taxi/test.txt"
    )
}

SAMPLE_SIZES = [10, 100, 1000]

# --- Template with Dynamic Memory ---
SH_TEMPLATE = """#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name={job_id}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output={job_id}.log
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem={mem_val}
#
    
# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Experiment: {job_id}"
echo "----------------------------------------------------"

export PYTHONNOUSERSITE=1
source {conda_sh}
conda activate ml_project

cd {proj_dir}

python model_experiment.py \\
    --source_dir {src_dir} \\
    --target_dir {tgt_dir} \\
    --source_list_path {src_list} \\
    --target_list_path {tgt_list} \\
    --source_test_list_path {src_test_list} \\
    --src_samples {n} \\
    --tgt_samples {n} \\
    --ratio_src_samples 0 \\
    --ratio_tgt_samples {n} \\
    --num_runs 100 \\
    --block_idx 4 \\
    --seed_base 32 \\
    --batch_size 4096 \\
    --modelStr "{model_str}" \\
    --file_name "{job_id}.json"

echo "----------------------------------------------------"
echo "Job finished: $(date)"
echo "----------------------------------------------------"
"""

def main():
    sbatch_commands = []

    for model_folder, model_str in MODELS.items():
        for n in SAMPLE_SIZES:
            # Memory Logic
            # N=10, 100 -> 64G | N=1000 -> 128G
            mem_val = "128G" if n == 1000 else "64G"
            
            exp_output_dir = os.path.join(BASE_ROOT, model_folder, str(n))
            os.makedirs(exp_output_dir, exist_ok=True)

            for src_label, (src_dir, src_list, src_test_list) in DATA_MAP.items():
                for tgt_label, (tgt_dir, tgt_list, _) in DATA_MAP.items():
                    
                    # File naming per your request: Dataset_ModelM#Samples.sh
                    # Added '2' (to) in middle to ensure CULane->Curvelanes doesn't overwrite CULane->CULane
                    job_id = f"{src_label}2{tgt_label}_{model_str}M{n}"
                    file_name = f"{src_label}2{tgt_label}_{model_str}M{n}.sh"
                    file_path = os.path.join(exp_output_dir, file_name)

                    content = SH_TEMPLATE.format(
                        job_id=job_id,
                        mem_val=mem_val,
                        conda_sh=CONDA_PROFILE,
                        proj_dir=PROJECT_DIR,
                        model_str=model_str,
                        src_label=src_label,
                        tgt_label=tgt_label,
                        src_dir=src_dir,
                        src_list=src_list,
                        src_test_list=src_test_list,
                        tgt_dir=tgt_dir,
                        tgt_list=tgt_list,
                        n=n
                    )

                    with open(file_path, "w") as f:
                        f.write(content)
                    
                    sbatch_commands.append(f"sbatch \"{file_path}\"")

    master_path = os.path.join(PROJECT_DIR, "run_all_permutations.sh")
    with open(master_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        # Adding a tiny sleep to avoid overwhelming the Slurm scheduler
        f.write("\n".join([f"{cmd}\nsleep 0.1" for cmd in sbatch_commands]))
    
    os.chmod(master_path, 0o755)
    print(f"Success! {len(sbatch_commands)} permutation scripts generated.")
    print(f"N=10/100: 64GB | N=1000: 128GB")

if __name__ == "__main__":
    main()
