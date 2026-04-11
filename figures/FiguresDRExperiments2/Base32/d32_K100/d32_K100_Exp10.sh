#!/bin/bash
#SBATCH --job-name=d32_K100_E10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=d32_K100_Exp10.log
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

echo "----------------------------------------------------"
echo "Recovering Job: d32 | K=100 | Exp=10"
echo "----------------------------------------------------"

export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh
conda activate ml_project
cd /home1/adoyle2025/Distribution-Shift-Lane-Perception

python shift_concat_experiment.py \
    --source_dir /home1/adoyle2025/Datasets/Datasets/CULane \
    --target_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes \
    --source_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt \
    --target_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \
    --source_test_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/test.txt \
    --src_samples 100 \
    --tgt_samples 100 \
    --ratio_src_samples 90 \
    --ratio_tgt_samples 10 \
    --num_runs 100 \
    --block_idx 4 \
    --seed_base 32 \
    --batch_size 512 \
    --dConfig "d32" \
    --save_all_image_paths True \
    --file_name "d32_K100.json"

echo "Job finished: $(date)"
