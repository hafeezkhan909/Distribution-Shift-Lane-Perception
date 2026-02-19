#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=d64ids_K10_E6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=d64ids_K10_Exp6.log
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start Time: $(date)"
echo "Config: d64ids | Samples: 10"
echo "Experiment 6: Src=50% (5), Tgt=50% (5)"
echo "----------------------------------------------------"

export PYTHONNOUSERSITE=1

echo "Initializing conda for script..."
source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh

conda activate ml_project

echo "Conda environment activated and isolated:"
conda info --env

cd /home1/adoyle2025/Distribution-Shift-Lane-Perception

echo "Starting Unit Test..."

python shift_concat_experiment.py \
    --source_dir /home1/adoyle2025/Datasets/Datasets/CULane \
    --target_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes \
    --source_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt \
    --target_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \
    --source_test_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/test.txt \
    --src_samples 10 \
    --tgt_samples 10 \
    --ratio_src_samples 5 \
    --ratio_tgt_samples 5 \
    --num_runs 100 \
    --block_idx 4 \
    --seed_base 32 \
    --batch_size 2048 \
    --dConfig "d64ids" \
    --save_all_image_paths True \
    --file_name "d64ids_K10.json"

echo "----------------------------------------------------"
echo "Job finished: $(date)"
echo "----------------------------------------------------"
