#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=P2100Samples_ImageNetModel_AssistTaxiData
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=P2100Samples_ImageNetModel_AssistTaxiData.log
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#
    
# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Experiment: P2100Samples_ImageNetModel_AssistTaxiData"
echo "----------------------------------------------------"

export PYTHONNOUSERSITE=1
source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh
conda activate ml_project

cd /home1/adoyle2025/Distribution-Shift-Lane-Perception

python model_experimentP2.py \
    --source_dir /home1/adoyle2025/Datasets/Datasets/ASSIST-Taxi \
    --target_dir /home1/adoyle2025/Datasets/Datasets/ASSIST-Taxi \
    --source_list_path /home1/adoyle2025/Datasets/Datasets/ASSIST-Taxi/train.txt \
    --target_list_path /home1/adoyle2025/Datasets/Datasets/ASSIST-Taxi/train.txt \
    --source_test_list_path /home1/adoyle2025/Datasets/Datasets/ASSIST-Taxi/test.txt \
    --src_samples 100 \
    --tgt_samples 100 \
    --ratio_src_samples 0 \
    --ratio_tgt_samples 100 \
    --num_runs 100 \
    --block_idx 4 \
    --seed_base 32 \
    --batch_size 64 \
    --modelStr "ImageNet" \
    --file_name "P2100Samples_ImageNetModel_AssistTaxiData.json"

echo "----------------------------------------------------"
echo "Job finished: $(date)"
echo "----------------------------------------------------"
