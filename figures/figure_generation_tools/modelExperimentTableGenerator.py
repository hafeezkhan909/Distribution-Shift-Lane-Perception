import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration & Paths ---
LOG_DIR = "/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs"
BASE_OUTPUT_DIR = (
    "/home1/adoyle2025/Distribution-Shift-Lane-Perception/ModelExperimentFigures"
)

# Define the matrix axes based on actual modelStr values used in experiments
# REMOVED: "ASSIST_Taxi" and "AssistTaxi"
MODELS = ["ImageNet", "Random", "CU_Lane", "CurveLanes"]
DATASETS = ["CULanes", "Curvelanes"]
SAMPLE_SIZES = [10, 100, 1000]
PHASES = [1, 2]


def extract_tpr_from_json(filepath):
    """Safely extracts TPR from the specific nested JSON structure."""
    if not os.path.exists(filepath):
        return "In Progress"

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            tpr_value = data["experiments"][0]["data"]["Data Shift Test Data"]["TPR"]
            return float(tpr_value)
    except (json.JSONDecodeError, KeyError, IndexError, ValueError):
        return "In Progress"


def main():
    sns.set_theme(style="whitegrid")

    row_labels = []
    for src in DATASETS:
        for tgt in DATASETS:
            row_labels.append(f"{src} \u2192 {tgt}")

    for phase in PHASES:
        phase_output_dir = os.path.join(BASE_OUTPUT_DIR, f"Phase{phase}")
        os.makedirs(phase_output_dir, exist_ok=True)

        for n in SAMPLE_SIZES:
            print(f"Processing Phase {phase} - {n} Samples...")

            results_grid = {
                model: {row: "In Progress" for row in row_labels} for model in MODELS
            }

            for model in MODELS:
                for src in DATASETS:
                    for tgt in DATASETS:

                        # --- FILENAME GENERATION LOGIC ---
                        if phase == 1:
                            if src == tgt:
                                filename = f"{n}Samples_{model}Model_{src}Data.json"
                            else:
                                filename = (
                                    f"{n}Samples_{model}Model_{src}2{tgt}Data.json"
                                )

                        elif phase == 2:
                            if src == tgt:
                                filename = f"P2{n}Samples_{model}Model_{src}Data.json"
                            else:
                                filename = (
                                    f"P2{n}Samples_{model}Model_{src}2{tgt}Data.json"
                                )
                        # -------------------------------------------

                        filepath = os.path.join(LOG_DIR, filename)
                        row_key = f"{src} \u2192 {tgt}"
                        tpr = extract_tpr_from_json(filepath)
                        results_grid[model][row_key] = tpr

            df = pd.DataFrame(results_grid)
            df = df.reindex(index=row_labels, columns=MODELS)

            # --- Generate LaTeX Table ---
            latex_df = df.copy()
            for col in latex_df.columns:
                latex_df[col] = latex_df[col].apply(
                    lambda x: f"{x:.2f}" if isinstance(x, float) else x
                )

            try:
                latex_str = latex_df.style.to_latex(
                    caption=f"Phase {phase} TPR (\\%) Results across Distribution Shifts - {n} Samples",
                    label=f"tab:phase{phase}_tpr_{n}",
                    hrules=True,
                )
            except AttributeError:
                latex_str = latex_df.to_latex(
                    caption=f"Phase {phase} TPR (\\%) Results across Distribution Shifts - {n} Samples",
                    label=f"tab:phase{phase}_tpr_{n}",
                )

            tex_filepath = os.path.join(
                phase_output_dir, f"Table_Phase{phase}_{n}Samples.tex"
            )
            with open(tex_filepath, "w") as f:
                f.write(latex_str)
            print(f"  -> Saved LaTeX Table: {tex_filepath}")

            # --- Generate Figure (Heatmap) ---
            df_numeric = df.replace("In Progress", np.nan).apply(pd.to_numeric)

            # Adjusted figure size for the smaller 4x4 grid
            plt.figure(figsize=(10, 7))

            ax = sns.heatmap(
                df_numeric,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                vmin=0,
                vmax=100,
                linewidths=0.5,
                cbar_kws={"label": "True Positive Rate (TPR %)"},
            )

            plt.title(
                f"Phase {phase} Distribution Shift: TPR (%) - {n} Samples\n(Blank cells indicate 'In Progress')",
                fontsize=14,
            )
            plt.xlabel("Model Pretraining Weights", fontsize=12)
            plt.ylabel("Evaluation (Source \u2192 Target)", fontsize=12)

            plt.yticks(rotation=0)
            plt.tight_layout()

            fig_filepath = os.path.join(
                phase_output_dir, f"Figure_Phase{phase}_{n}Samples.png"
            )
            plt.savefig(fig_filepath, dpi=300)
            plt.close()
            print(f"  -> Saved Heatmap Figure: {fig_filepath}")


if __name__ == "__main__":
    main()
