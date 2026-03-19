#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=P2100Samples_CurveLanesModel_CULanes2CurvelanesData
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=P2100Samples_CurveLanesModel_CULanes2CurvelanesData.log
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#
    
# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Experiment: P2100Samples_CurveLanesModel_CULanes2CurvelanesData"
echo "----------------------------------------------------"

export PYTHONNOUSERSITE=1
source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh
conda activate ml_project

cd /home1/adoyle2025/Distribution-Shift-Lane-Perception

python model_experimentP2.py \
    --source_dir /home1/adoyle2025/Datasets/Datasets/CULane \
    --target_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes \
    --source_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt \
    --target_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \
    --source_test_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/test.txt \
    --src_samples 100 \
    --tgt_samples 100 \
    --ratio_src_samples 0 \
    --ratio_tgt_samples 100 \
    --num_runs 100 \
    --block_idx 4 \
    --seed_base 32 \
    --batch_size 64 \
    --modelStr "CurveLanes" \
    --file_name "P2100Samples_CurveLanesModel_CULanes2CurvelanesData.json"

echo "----------------------------------------------------"
echo "Job finished: $(date)"
echo "----------------------------------------------------"
