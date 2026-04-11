#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=B1MMDAggCU_Lane
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output="/home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash/MMDAgg/CULaneTrain2CULaneTrainDistill100.log"
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-user=adoyle2025@my.fit.edu
#SBATCH --mail-type=END,FAIL

# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: \$SLURM_JOB_ID"
echo "Running on host: \$(hostname)"
echo "Experiment: CULaneTrain2CULaneTrainDistill100"
echo "----------------------------------------------------"

export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export TF_CPP_MIN_LOG_LEVEL=3

source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh

conda activate ml_project

python -m pip install "numpy<2.0.0"

cd /home1/adoyle2025/Distribution-Shift-Lane-Perception

python model_experimentP2.py \
    --source_dir /home1/adoyle2025/Datasets/Datasets/CULane \
    --target_dir /home1/adoyle2025/Datasets/Datasets/CULane \
    --source_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt \
    --target_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt \
    --src_samples 100 \
    --tgt_samples 100 \
    --num_runs 100 \
    --num_calib 100 \
    --block_idx 4 \
    --seed_base 32 \
    --batch_size 64 \
    --latent_dim 32 \
    --test_type "MMDAgg" \
    --permutation_test_iterations 0 \
    --file_location "/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs/ModelExperiments/P2" \
    --file_name "CULaneTrain2CULaneTrainDistill100.json" \
    --modelStr "DISTILL"

echo "----------------------------------------------------"
echo "Job finished: \$(date)"
echo "----------------------------------------------------"