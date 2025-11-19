# Distribution-Shift-Lane-Perception

## Overview

The pipeline estimates an empirical MMD threshold (`τ`) using *same-domain calibration*, and then tests whether samples from another dataset (or perturbed version of the same dataset) come from a statistically different distribution.

The pipeline consists of:

1. **Calibration (same-domain):** Establishes a "null distribution" of MMD values by repeatedly comparing the source data to random subsets of itself. This determines the threshold `τ` (tau) for shift detection.

2. **Testing (cross-domain or shifted):** Compares the reference features with samples from another dataset (e.g., CULane) or a synthetically shifted version of the same dataset.

3. **Evaluation:** If the MMD statistic exceeds `τ`, a significant distribution shift is detected.

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

## Files

### `run_experiment.py`

Main experiment pipeline.

* Loads a pretrained (ResNet) convolutional autoencoder (`ConvAutoencoderFC`) for feature extraction.

* Calibrates MMD threshold using same-domain (e.g., Curvelanes → Curvelanes) comparisons.

* Tests cross-domain shifts (e.g., Curvelanes → CULane) using the calibrated threshold.

* Saves extracted features, calibration results, and per-run statistics in the `features/` directory.

### `mmd_test.py`

Implements the **Maximum Mean Discrepancy (MMD)** test using `torch_two_sample`.
It computes both the MMD statistic and bootstrap-based p-value between two feature distributions.

```
def mmd_test(X_src, X_tgt):
    \"\"\"
    Args:
        X_src (np.ndarray): Source domain features, shape (N, D)
        X_tgt (np.ndarray): Target domain features, shape (M, D)
    Returns:
        (mmd_statistic, p_value)
    \"\"\"


```

The kernel bandwidth is set using the median pairwise distance heuristic (1 / median_dist), consistent with prior works.

## 📋 Prerequisites

### 1. Python 3 and required packages:

   > Notice: Python 3.10.19 is **Highly Reccomended**

   ```
   # Instally numpy first because it is a torch_two_sample dependency
   pip install numpy<2.0

   # Install remaining dependencies
   pip install scipy torch torchvision tqdm Pillow && pip install ./torch_two_sample
   ```

### 2.  Dataset Structure
All datasets must adhere to a simple file-list structure for our data loaders:

**Root Directory (root_dir):** The absolute path to the base folder containing all image files.

**List File (list_path):** A text file (e.g., train.txt) containing paths to images, one per line.

The paths listed inside the List File must be relative to the root_dir.

**Example**
Given the root directory is datasets/MyProject:

|    File Path in System    |   Path in `train.txt`  |
| --------------------------------- | -------------- |
| `datasets/MyProject/data/001.jpg` | `data/001.jpg` |
| `datasets/MyProject/data/002.jpg` | `data/002.jpg` |

## 🚀 How to Run `shift_experiment.py`

Execute the script via the command line. The experiment performs feature extraction, MMD calibration, a sanity check, and finally the data shift test.

### Basic Usage

```bash
python shift_experiment.py --source CULane --target Curvelanes --shift <SHIFT_TYPE> [SHIFT_ARGS]
```

### Shift Scenarios

Select a shift type and provide the corresponding magnitude argument:

| Shift Type | Required Argument | Example Command |
| :--- | :--- | :--- |
| **Gaussian Noise** | `--std` | `python shift_experiment.py --shift gaussian --std 1.5` |
| **Rotation** | `--rotation_angle` | `python shift_experiment.py --shift rotation_shift --rotation_angle 30` |
| **Translation** | `--width/height_shift_frac` | `python shift_experiment.py --shift translation_shift --width_shift_frac 0.2` |
| **Shear** | `--shear_angle` | `python shift_experiment.py --shift shear_shift --shear_angle 15` |
| **Zoom** | `--zoom_factor` | `python shift_experiment.py --shift zoom_shift --zoom_factor 1.3` |
| **Flips** | *(None)* | `python shift_experiment.py --shift horizontal_flip_shift` |

### Common Optional Flags

  * `--source` / `--target`: Dataset names (Default: `CULane`, `Curvelanes`).
  * `--src_samples` / `--tgt_samples`: Number of samples to process (Default: `1000`, `100`).
  * `--num_runs`: Number of times to repeat the shift test (Default: `10`).
  * `--file_name`: Name of the output JSON log (Default: `sanity_check.json`).
  * `--cropImg`: Boolean flag to crop images (Default: `False`).

### Output

Results, including MMD scores, TPR (True Positive Rate), and pass/fail status, are printed to the console and saved to `logs/<file_name>`.

## 🚀 How to Run `run_experiment.py`

The script `run_experiment.py` is configured to run from the command line.

### Basic Execution

To run the experiment with all default settings (comparing 1000 samples of `Curvelanes` to 100 samples of `Curvelanes`):

```
python run_experiment.py


```

### Checking All Arguments

To see a full list of all available commands, their defaults, and their descriptions:

```
python run_experiment.py -h


```

### Example: Cross-Domain Test

Here is a more practical example. Let's test for a shift between the `Curvelanes` **training set** and the `CULane` **test set**.

We will use:

* `Curvelanes` as the source (`-s`)

* `CULane` as the target (`-t`)

* `train` split for the source (`-p`)

* `test` split for the target (`-g`)

* `5000` source samples (`-r`)

* `500` target/calibration samples (`-a`)

* A different seed (`--seed_base`)

```
python run_experiment.py \
    -s Curvelanes \
    -t CULane \
    -p train \
    -g test \
    -r 5000 \
    -a 500 \
    --seed_base 123


```

## ⚙️ Command-Line Arguments

Here is a detailed list of all available arguments, based on the script's `argparse` setup.

### Dataset Arguments

| Flag (Short) | Flag (Long) | Default | Description | 
 | ----- | ----- | ----- | ----- | 
| `-s` | `--source` | `"Curvelanes"` | Source dataset name (e.g., `Curvelanes`, `CULane`). | 
| `-t` | `--target` | `"Curvelanes"` | Target dataset name for the sanity check. | 
| `-p` | `--src_split` | `"train"` | Source dataset split (e.g., `train`, `test`). | 
| `-g` | `--tgt_split` | `"train"` | Target dataset split. | 

### Sampling Arguments

| Flag (Short) | Flag (Long) | Default | Description | 
 | ----- | ----- | ----- | ----- | 
| `-r` | `--src_samples` | `1000` | Number of samples for the source reference. | 
| `-a` | `--tgt_samples` | `100` | Number of samples for each target test run. | 
| `-b` | `--block_idx` | `0` | Block index for chunked source loading (e.g., `0` for samples 0-999, `1` for 1000-1999). | 

### Model & MMD Test Arguments

| Flag (Short) | Flag (Long) | Default | Description | 
 | ----- | ----- | ----- | ----- | 
| `-i` | `--batch_size` | `16` | Batch size for feature extraction. | 
| `-z` | `--image_size` | `512` | Image resize dimension (e.g., 512x512). | 
| `-e` | `--num_calib` | `100` | Number of calibration runs to build the null distribution. | 
| `-n` | `--alpha` | `0.05` | Significance level for the test (e.g., 0.05 for 95% confidence). | 

### Reproducibility

| Flag (Short) | Flag (Long) | Default | Description | 
 | ----- | ----- | ----- | ----- | 
| (None) | `--seed_base` | `42` | Base seed for all random sampling. | 

## Outputs

All feature and result files are stored in the `features/` directory:

```
features/
 ├─ Curvelanes_train_1000_0.npy   (Cached features for the source)
 ├─ calibration_null_mmd.npy      (MMD values from calibration runs)
 ├─ mmd_curvelanes_100runs.npy    (MMD values from cross-domain test runs)
 └─ tpr_curvelanes_100runs.npy    (Detection results (0 or 1) for each test run)


```