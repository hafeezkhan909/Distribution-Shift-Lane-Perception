#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=graph
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=graph.log
#SBATCH --partition=eternity
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
    
# --- Job Execution ---
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

echo "Starting Unit Test..."

python3 visGen2.py \
  --log-path /home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash/DRExperiments \
  --sh-path  /home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash/DRExperiments \
  --recursive \
  --output-dir figures_organized \
  --expect-groups 24

echo "Done!"