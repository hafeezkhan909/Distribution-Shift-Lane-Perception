#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=DistillTrainCULane
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=DistillTrainCULane.log
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

echo "----------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start Time: $(date)"
echo "----------------------------------------------------"

# --- GPU Safety Flags ---
# Prevents JAX/XLA from pre-allocating all VRAM and blocking PyTorch
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONNOUSERSITE=1
# Recommended for A100 clusters to prevent NCCL handshake hangs
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


echo "Initializing conda for script..."
source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh
conda activate clrernet


echo "Conda environment activated and isolated:"
conda info --env

# Move to the root of the perception project
cd /home1/adoyle2025/Distribution-Shift-Lane-Perception

echo "Starting Distillation Training..."

# Execute the NEW distillation training script
# Note: Ensure the path to teacher_weights is correct for your CULane checkpoint
python models/trainingScripts/trainDistil.py \
    --dataset_name "CULane" \
    --dataset_dir "/home1/adoyle2025/Datasets/Datasets/CULane" \
    --dataset_list "/home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt" \
    --teacher_config "configs/clrernet/culane/clrernet_culane_dla34.py" \
    --teacher_weights "/home1/adoyle2025/CLRerNet-Runtime-Monitor-for-Lane-Detection/work_dirs/clrernet_culane_dla34/epoch_15.pth" \
    --distill_weight 0.7 \
    --samples 100000 \
    --batch_size 32 \
    --image_size 512 \
    --epochs 80 \
    --learning_rate 5e-5

echo "----------------------------------------------------"
echo "Job finished: $(date)"
echo "----------------------------------------------------"
