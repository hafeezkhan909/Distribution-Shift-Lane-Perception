#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=bashGen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=bashGen.log
#SBATCH --partition=eternity
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#
    
# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start Time: $(date)"
echo "----------------------------------------------------"

# Configuration
K=20  # Total samples (can be changed to any multiple of 10)
srcSamples=1000
SCRIPT_PREFIX="a32ttw"
OUTPUT_DIR="/home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash/asym32dthousandTwenty"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Generating ratio experiment scripts..."
echo "Total samples (k): $K"
echo "========================================"

# Calculate number of experiments (0% to 100% in 10% steps = 11 experiments)
exp_num=1

# Iterate from 0% to 100% source samples (in 10% increments)
for src_percent in {0..100..10}; do
    # Calculate target percent (complement to 100%)
    tgt_percent=$((100 - src_percent))
    
    # Calculate actual sample counts
    ratio_src=$((K * src_percent / 100))
    ratio_tgt=$((K * tgt_percent / 100))
    
    # Define script filename
    script_name="${OUTPUT_DIR}/${SCRIPT_PREFIX}${exp_num}.sh"
    log_name="${SCRIPT_PREFIX}${exp_num}.log"
    job_name="${SCRIPT_PREFIX}${exp_num}"
    
    echo "Creating Experiment ${exp_num}: Src=${src_percent}% (${ratio_src}), Tgt=${tgt_percent}% (${ratio_tgt})"
    
    # Generate the script
    cat > "$script_name" << EOF
#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=${job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=${log_name}
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#
    
# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: \$SLURM_JOB_ID"
echo "Running on host: \$(hostname)"
echo "Start Time: \$(date)"
echo "Experiment ${exp_num}: Src=${src_percent}% (${ratio_src} samples), Tgt=${tgt_percent}% (${ratio_tgt} samples)"
echo "----------------------------------------------------"


export PYTHONNOUSERSITE=1

echo "Initializing conda for script..."
source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh

conda activate ml_project

echo "Conda environment activated and isolated:"
conda info --env

cd /home1/adoyle2025/Distribution-Shift-Lane-Perception

echo "Starting Unit Test..."

python shift_concat_experiment.py \\
    --source_dir /home1/adoyle2025/Datasets/Datasets/CULane \\
    --target_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes \\
    --source_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt \\
    --target_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \\
    --source_test_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/test.txt \\
    --src_samples ${srcSamples} \\
    --tgt_samples ${K} \\
    --ratio_src_samples ${ratio_src} \\
    --ratio_tgt_samples ${ratio_tgt} \\
    --num_runs 100 \\
    --block_idx 4 \\
    --seed_base 32 \\
    --batch_size 4096 \\
    --thirty_two_dimensional True \\
    --save_all_image_paths True \\
    --file_name "${SCRIPT_PREFIX}.json"

echo "----------------------------------------------------"
echo "Job finished: \$(date)"
echo "----------------------------------------------------"
EOF
    
    # Increment experiment number
    exp_num=$((exp_num + 1))
done

echo "========================================"
echo "Generated $((exp_num - 1)) scripts successfully!"
echo ""
echo "Generated files:"
ls -1 ${OUTPUT_DIR}/${SCRIPT_PREFIX}*.sh
echo ""
echo "To submit all jobs:"
echo "  for script in ${SCRIPT_PREFIX}*.sh; do sbatch \$script; done"
echo ""
echo "To submit individually:"
for i in $(seq 1 $((exp_num - 1))); do
    echo "  sbatch ${SCRIPT_PREFIX}${i}.sh"
done
