#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=MasterGen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=master_gen.log
#SBATCH --partition=eternity
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB

# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start Time: $(date)"
echo "----------------------------------------------------"

# Base Path
BASE_DIR="/home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash/DRExperiments2"

# 1. Define the lists to iterate over
configs=("d128rel" "d64rel" "d32rel" "d128gdd" "d64gdd" "d32gdd" "d64ids" "d128ids" "d32")
sample_sizes=(10 100 1000)

# 2. Iterate over Configs
for config in "${configs[@]}"; do
    
    # Determine parent folder name based on config type for organization
    case "$config" in
        *"ids"*) type_dir="IncreaseDimensionality" ;;
        *"rel"*) type_dir="RemoveExtraLayer" ;;
        *"gdd"*) type_dir="GraduallyDecreaseDimensions" ;;
        "d32")   type_dir="Base32" ;;
        *)       type_dir="Misc" ;;
    esac

    # 3. Iterate over Sample Sizes (K)
    for K in "${sample_sizes[@]}"; do
        
        # Setup variables for this batch
        srcSamples=$K
        SCRIPT_PREFIX="${config}"
        
        # Create a unique output directory: LocalBash/Type/Config_Size
        # Example: LocalBash/IncreaseDimensionality/d128ids_K10
        OUTPUT_DIR="${BASE_DIR}/${type_dir}/${config}_K${K}"
        mkdir -p "$OUTPUT_DIR"

        echo "========================================"
        echo "Generating scripts for: $config | K=$K"
        echo "Output Dir: $OUTPUT_DIR"
        echo "========================================"

        # Inner Loop: Generate the 11 Ratio Experiments (0% to 100%)
        exp_num=1
        
        for src_percent in {0..100..10}; do
            # Calculate target percent (complement to 100%)
            tgt_percent=$((100 - src_percent))
            
            # Calculate actual sample counts
            ratio_src=$((K * src_percent / 100))
            ratio_tgt=$((K * tgt_percent / 100))
            
            # Define file names
            # Using K${K} in filename ensures uniqueness if files are moved later
            script_name="${OUTPUT_DIR}/${SCRIPT_PREFIX}_K${K}_Exp${exp_num}.sh"
            log_name="${SCRIPT_PREFIX}_K${K}_Exp${exp_num}.log"
            job_name="${SCRIPT_PREFIX}_K${K}_E${exp_num}"
            
            # Generate the script content
            cat > "$script_name" << EOF
#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=${job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=${log_name}
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: \$SLURM_JOB_ID"
echo "Running on host: \$(hostname)"
echo "Start Time: \$(date)"
echo "Config: ${config} | Samples: ${K}"
echo "Experiment ${exp_num}: Src=${src_percent}% (${ratio_src}), Tgt=${tgt_percent}% (${ratio_tgt})"
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
    --batch_size 512 \\
    --dConfig "${SCRIPT_PREFIX}" \\
    --save_all_image_paths True \\
    --file_name "${SCRIPT_PREFIX}_K${K}.json"

echo "----------------------------------------------------"
echo "Job finished: \$(date)"
echo "----------------------------------------------------"
EOF
            
            # Make executable (optional but good practice)
            chmod +x "$script_name"
            
            exp_num=$((exp_num + 1))
        done
        
        echo "Generated $((exp_num - 1)) scripts in $OUTPUT_DIR"
    done
done

echo "----------------------------------------------------"
echo "All Generations Complete."
echo "To run everything (WARNING: ~300 jobs):"
echo "  find ${BASE_DIR} -name \"*.sh\" -exec sbatch {} \;"
echo "----------------------------------------------------"