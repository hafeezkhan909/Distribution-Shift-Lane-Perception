import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration & Paths ---
LOG_DIR = "/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs"
OUTPUT_DIR = "/home1/adoyle2025/Distribution-Shift-Lane-Perception/ModelExperimentFigures/Phase1"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the matrix axes based on actual modelStr values used in experiments
MODELS = ["ImageNet", "Random", "CU_Lane", "CurveLanes", "ASSIST_Taxi"]
DATASETS = ["CULanes", "Curvelanes", "AssistTaxi"]
SAMPLE_SIZES = [10, 100, 1000]

def extract_tpr_from_json(filepath):
    """Safely extracts TPR from the specific nested JSON structure."""
    if not os.path.exists(filepath):
        return "In Progress"
        
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            # Structure: "experiments" [{"data" {..., "Data Shift Test Data": {... "TPR": 3.0,...}}}]
            tpr_value = data["experiments"][0]["data"]["Data Shift Test Data"]["TPR"]
            return float(tpr_value)
    except (json.JSONDecodeError, KeyError, IndexError, ValueError):
        # Handle partially written, empty, or malformed JSONs as "In Progress"
        return "In Progress"

def main():
    # Set a cleaner style for plots
    sns.set_theme(style="whitegrid")

    for n in SAMPLE_SIZES:
        print(f"Processing Phase 1 - {n} Samples...")
        
        # 1. Initialize empty results grid (Datasets as Rows, Models as Columns)
        # We use a nested dictionary structure for easy conversion to Pandas
        results_grid = {model: {dataset: "In Progress" for dataset in DATASETS} for model in MODELS}
        
        # 2. Parse Logs and populate grid
        for model in MODELS:
            for dataset in DATASETS:
                filename = f"{n}Samples_{model}Model_{dataset}Data.json"
                filepath = os.path.join(LOG_DIR, filename)
                
                tpr = extract_tpr_from_json(filepath)
                results_grid[model][dataset] = tpr
        
        # 3. Convert to Pandas DataFrame (Rows=Datasets, Columns=Models)
        df = pd.DataFrame(results_grid)
        # Ensure the order matches our defined lists
        df = df.reindex(index=DATASETS, columns=MODELS)
        
        # --- 4. Generate LaTeX Table ---
        # Using string representation formatting to handle "In Progress" strings alongside numbers
        latex_df = df.copy()
        for col in latex_df.columns:
            latex_df[col] = latex_df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)

        try:
            # Modern Pandas (2.0+) standard
            latex_str = latex_df.style.to_latex(
                caption=f"Phase 1 TPR (%) Results - {n} Samples", 
                label=f"tab:tpr_{n}",
                hrules=True
            )
        except AttributeError:
            # Fallback for older Pandas versions
            latex_str = latex_df.to_latex(
                caption=f"Phase 1 TPR (%) Results - {n} Samples", 
                label=f"tab:tpr_{n}"
            )
            
        tex_filepath = os.path.join(OUTPUT_DIR, f"Table_Phase1_{n}Samples.tex")
        with open(tex_filepath, "w") as f:
            f.write(latex_str)
        print(f"  -> Saved LaTeX Table: {tex_filepath}")
        
        # --- 5. Generate Figure (Heatmap) ---
        # Create a strictly numeric copy for plotting. Replace "In Progress" with NaN.
        # Seaborn automatically leaves NaN cells empty/blank in the heatmap.
        df_numeric = df.replace("In Progress", np.nan).apply(pd.to_numeric)
        
        plt.figure(figsize=(12, 7))
        
        # --- UPDATE: Enforce 0-100 scale ---
        # vmin=0, vmax=100 restricts the colorbar range to valid percentages.
        ax = sns.heatmap(
            df_numeric, 
            annot=True,          # Write numbers in cells
            fmt=".2f",          # format to 2 decimal places
            cmap="viridis",      # Color scheme
            vmin=0,             # Explicit minimum scale
            vmax=100,            # Explicit maximum scale
            linewidths=.5,       # Add gridlines between cells
            cbar_kws={'label': 'True Positive Rate (TPR %)'} # Colorbar label
        )
        
        plt.title(f"Phase 1 Distribution Shift: TPR (%) - {n} Samples\n(Blank cells indicate 'In Progress')", fontsize=14)
        plt.xlabel("Model Pretraining Weights", fontsize=12)
        plt.ylabel("Evaluation Dataset", fontsize=12)
        plt.tight_layout()
        
        fig_filepath = os.path.join(OUTPUT_DIR, f"Figure_Phase1_{n}Samples.png")
        plt.savefig(fig_filepath, dpi=300) # High resolution for publications
        plt.close()
        print(f"  -> Saved Heatmap Figure: {fig_filepath}")

if __name__ == "__main__":
    main()
