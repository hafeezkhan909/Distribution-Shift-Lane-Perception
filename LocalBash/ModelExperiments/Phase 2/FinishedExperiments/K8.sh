#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=H2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output="/home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash/BKS/P2CULaneTrain2CULaneTest100ImageNet.log"
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-user=adoyle2025@my.fit.edu
#SBATCH --mail-type=END,FAIL

# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: \$SLURM_JOB_ID"
echo "Running on host: Florida Tech HPC"
echo "----------------------------------------------------"

export PYTHONNOUSERSITE=1

source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh

conda activate ml_project

cd /home1/adoyle2025/Distribution-Shift-Lane-Perception

python model_experimentP2.py \
    --source_dir /home1/adoyle2025/Datasets/Datasets/CULane \
    --target_dir /home1/adoyle2025/Datasets/Datasets/CULane \
    --source_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt \
    --target_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/test.txt \
    --src_samples 100 \
    --tgt_samples 100 \
    --num_runs 100 \
    --num_calib 100 \
    --block_idx 4 \
    --seed_base 32 \
    --batch_size 64 \
    --latent_dim 32 \
    --test_type "BKS" \
    --permutation_test_iterations 0 \
    --file_location "/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs/ModelExperiments/P2" \
    --file_name "P2CULaneTrain2CULaneTest100ImageNet.json" \
    --modelStr "ImageNet"

echo "----------------------------------------------------"
echo "Job finished: \$(date)"
echo "----------------------------------------------------"