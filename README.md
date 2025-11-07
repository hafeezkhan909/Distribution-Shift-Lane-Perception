# Distribution-Shift-Lane-Perception

Only two files are required to run the pipeline:

* `run_experiment.py`
* `mmd_test.py`

All other files or helper scripts can be ignored.

---

## Overview

The pipeline estimates an empirical MMD threshold (`τ`) using *same-domain calibration*, and then tests whether samples from another dataset (or perturbed version of the same dataset) come from a statistically different distribution.

---

## How It Works

### 1. Calibration (same-domain)

The script first samples multiple subsets from the **same dataset** (e.g., Curvelanes) and computes the MMD statistic between each subset and a fixed reference set.
These values form the *null distribution* of MMD under no shift, from which the threshold `τ` is computed at the desired significance level (α = 0.05 by default).

### 2. Cross-domain (or shifted) testing

After calibration, the script compares the reference features with samples from another dataset (e.g., CULane) or a synthetically shifted version of the same dataset.
If the MMD statistic exceeds `τ`, a significant distribution shift is detected.

### 3. Aggregated evaluation

To measure stability, the script repeats the cross-domain test for 100 different random subsets and reports:

* Mean and standard deviation of MMD values
* **True Positive Rate (TPR)** — percentage of runs correctly detecting the shift

---

## Files

### `run_experiment.py`

Main experiment pipeline.

* Loads a pretrained (ResNet) convolutional autoencoder (`ConvAutoencoderFC`) for feature extraction.
* Calibrates MMD threshold using same-domain (e.g., Curvelanes → Curvelanes) comparisons.
* Tests cross-domain shifts (e.g., Curvelanes → CULane) using the calibrated threshold.
* Saves extracted features, calibration results, and per-run statistics in the `features/` directory.

You can later modularize the code to enable entering inputs through command line for any dataset to calibrate and evaluate different types of shifts.

### `mmd_test.py`

Implements the **Maximum Mean Discrepancy (MMD)** test using `torch_two_sample`.
It computes both the MMD statistic and bootstrap-based p-value between two feature distributions.

```python
def mmd_test(X_src, X_tgt):
    """
    Args:
        X_src (np.ndarray): Source domain features, shape (N, D)
        X_tgt (np.ndarray): Target domain features, shape (M, D)
    Returns:
        (mmd_statistic, p_value)
    """
```

The kernel bandwidth is set using the median pairwise distance heuristic (1 / median_dist), consistent with prior works.

Example:

```bash
python mmd_test.py
```

Output:

```
MMD statistic: 0.5123, p-value: 0.000001
✅ Significant shift detected.
```

---

## Running the Experiment

1. Place your datasets under `datasets/<DatasetName>/train/` and ensure the corresponding `train.txt` lists all image paths.
2. Run the full experiment:

   ```bash
   python run_experiment.py
   ```
3. The script automatically:

   * Extracts and saves features
   * Calibrates the null MMD distribution
   * Evaluates cross-domain detection and prints TPR

---

## Example Configuration

Inside `run_experiment.py`:

```python
source = "Curvelanes"     # Calibration dataset
target_cross = "CULane"   # Cross-domain test dataset
num_calib = 100           # Calibration iterations
num_runs = 100            # Cross-domain test runs
```

To test synthetic shifts within the same domain, replace `"CULane"` with your shifted dataset variant (e.g., `"Curvelanes_BrightnessShift"`).

---

## Outputs

All feature and result files are stored in:

```
features/
 ├─ Curvelanes_train_1000_0.npy
 ├─ calibration_null_mmd.npy
 ├─ mmd_curvelanes_100runs.npy
 └─ tpr_curvelanes_100runs.npy
```
