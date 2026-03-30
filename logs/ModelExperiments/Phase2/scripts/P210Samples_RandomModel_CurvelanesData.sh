#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=P210Samples_RandomModel_CurvelanesData
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs/Phase2_rerun/P210Samples_RandomModel_CurvelanesData.log
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Experiment: P210Samples_RandomModel_CurvelanesData"
echo "----------------------------------------------------"

export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh
conda activate ml_project
cd /home1/adoyle2025/Distribution-Shift-Lane-Perception

python model_experimentP2.py \
    --source_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes \
    --target_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes \
    --source_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \
    --target_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \
    --source_test_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \
    --src_samples 10 \
    --tgt_samples 10 \
    --ratio_src_samples 0 \
    --ratio_tgt_samples 10 \
    --num_runs 100 \
    --block_idx 4 \
    --seed_base 32 \
    --batch_size 64 \
    --modelStr "Random" \
    --file_location "/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs/Phase2_rerun" \
    --file_name "P210Samples_RandomModel_CurvelanesData.json"

echo "----------------------------------------------------"
echo "Job finished: $(date)"
echo "----------------------------------------------------"
