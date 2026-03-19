#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=P2TrainCurvelanes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=P2TrainCurvelanes.log
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#
echo "----------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start Time: $(date)"
echo "----------------------------------------------------"


export PYTHONNOUSERSITE=1

echo "Initializing conda for script..."
source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh

conda activate ml_project

echo "Conda environment activated and isolated:"
conda info --env

cd /home1/adoyle2025/Distribution-Shift-Lane-Perception

echo "Starting Model Training P2..."

# Execute the training script
python trainPhase2.py \
    --dataset_name "Curvelanes" \
    --dataset_dir "/home1/adoyle2025/Datasets/Datasets/Curvelanes" \
    --dataset_list "/home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt" \
    --samples 100000 \
    --batch_size 128 \
    --image_size 512 \
    --epochs 50 \
    --learning_rate 0.0001

echo "----------------------------------------------------"
echo "Job finished: $(date)"
echo "----------------------------------------------------"