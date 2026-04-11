#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=PKI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output="/home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash/KI.log"
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: \$SLURM_JOB_ID"
echo "Running on host: Florida Tech HPC"
echo "----------------------------------------------------"

export PYTHONNOUSERSITE=1

source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh

conda activate ml_project

cd /home1/adoyle2025/Distribution-Shift-Lane-Perception

python -m pip install jax jaxlib

echo "----------------------------------------------------"
echo "Job finished: \$(date)"
echo "----------------------------------------------------"