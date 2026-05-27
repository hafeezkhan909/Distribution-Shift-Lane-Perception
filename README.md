# Distribution-Shift-Lane-Perception

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
  - [Environment](#environment)
  - [Dataset Structure](#dataset-structure)
- [🚀 How to Run `experiment.py` (New Pipeline)](#-how-to-run-experimentpy-new-pipeline)
  - [Modes: `custom_weights` vs `imagenet_weights`](#modes-custom_weights-vs-imagenet_weights)
  - [Basic Usage](#basic-usage)
  - [Required Arguments](#required-arguments)
  - [Optional Configuration](#optional-configuration)
  - [Outputs](#outputs)
- [Other Scripts](#other-scripts)

## Overview

The pipeline estimates an empirical threshold (`τ`) using *same-domain calibration*, and then tests whether samples from another dataset (or a perturbed version of the same dataset) come from a statistically different distribution.

The pipeline consists of:

1. **Extract Features:** Encodes features from the source and target data and generates statistics from the features.

![Extract Features](figures/readMeGraphics/Extract%20Features.svg)

2. **Calibration:** Forms a null distribution of test statistics by comparing a fixed source reference set against many randomly sampled sets from the *same source domain*. `τ` is set as the `(1 - α)` percentile of this null distribution.

![Calibration Diagram](figures/readMeGraphics/Calibration.svg)

3. **Data Shift Test:** Compares the source reference features with samples from a target domain (e.g., CULane → Curvelanes) and reports the shift detection rate across runs.

![Data Shift Test Diagram](figures/readMeGraphics/Data%20Shift%20Test.svg)

---

## Setup

### Environment

#### Python 3 and Required Packages

> Notice: Python 3.8 and above

##### Two ways to setup (either or):

1. Conda Environment Setup
```
# Install everything using the environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate distribution_shift_perception

# Install the torch_two_sample dependency
git clone https://github.com/josipd/torch-two-sample.git
cd torch_two_sample
python setup.py install
```

2. Python Pip Setup
```
# Install dependencies
pip install scipy torch torchvision tqdm Pillow
pip install "numpy<2.0"

# Install the torch_two_sample dependency
git clone https://github.com/josipd/torch-two-sample.git
cd torch_two_sample
python setup.py install
```

### Dataset Structure

All datasets must adhere to a simple file-list structure for our data loaders:

**Root Directory (root_dir):** The absolute path to the base folder containing all image files.

**List File (list_path):** A text file (e.g., `train.txt`) containing paths to images, one per line.

The paths listed inside the List File must be relative to the `root_dir`.

**Example**

Given the root directory is `datasets/MyProject`:

| File Path in System | Path in `train.txt` |
| --- | --- |
| `datasets/MyProject/data/001.jpg` | `data/001.jpg` |
| `datasets/MyProject/data/002.jpg` | `data/002.jpg` |

---

## 🚀 How to Run `experiment.py` (New Pipeline)

`experiment.py` is the updated experiment runner. It:

- Extracts and **caches** latent encodings to disk (to avoid re-encoding on repeated runs).
- Runs the statistical tests **sequentially** (calibration → sanity check → shift test) to avoid PyTorch/JAX GPU memory collisions.
- Supports multiple two-sample tests:
  - **MMD**
  - **MMD_Agg**
  - **Energy**
  - **BKS**

### Modes: `custom_weights` vs `imagenet_weights`

`experiment.py` has two subcommands:

- `custom_weights`: uses a local weights directory for the autoencoder.
- `imagenet_weights`: uses ImageNet weights (no custom weights directory required).

### Basic Usage

> Tip: If you are on a 1-GPU machine, the script automatically forces **JAX to CPU** and keeps PyTorch on the GPU to prevent contention.

#### Example (ImageNet weights)

```bash
python experiment.py \
  --source_dir ./datasets/CULane \
  --target_dir ./datasets/Curvelanes \
  --source_list_path ./datasets/CULane/list/train.txt \
  --target_list_path ./datasets/Curvelanes/list/train.txt \
  imagenet_weights
```

#### Example (Custom weights)

```bash
python experiment.py \
  --source_dir ./datasets/CULane \
  --target_dir ./datasets/Curvelanes \
  --source_list_path ./datasets/CULane/list/train.txt \
  --target_list_path ./datasets/Curvelanes/list/train.txt \
  custom_weights \
  --model_weights_path ./weights/my_autoencoder_weights_dir
```

### Required Arguments

These are required regardless of mode:

| Argument | Description |
| :--- | :--- |
| `--source_dir` | Root directory for the Source dataset. |
| `--target_dir` | Root directory for the Target dataset. |
| `--target_list_path` | Path to the text file containing target image paths. |

> `--source_list_path` defaults to `./datasets/CULane/list/train.txt`, but you will almost always want to set it explicitly.

Mode-specific required args:

| Mode | Required Argument | Description |
| :--- | :--- | :--- |
| `custom_weights` | `--model_weights_path` | Path to the **directory** containing the custom weights. |
| `imagenet_weights` | *(none)* | Uses ImageNet weights. |

### Optional Configuration

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--sample_size` | `1000` | Number of samples per run (source + each sampled target/calibration set). |
| `--num_runs` | `100` | Number of target runs used to compute detection rate. |
| `--batch_size` | `128` | Batch size for feature extraction. |
| `--image_size` | `512` | Resize images to `image_size x image_size`. |
| `--alpha` | `0.05` | Significance level; `τ` is set at the `(1-α)` percentile. |
| `--seed_base` | `42` | Base seed for all random sampling. |
| `--permutation_test_iterations` | `1000` | Number of permutations used for MMD/MMD_Agg/Energy p-values. Set to `0` to skip p-values. |
| `--latent_dim` | `32` | Latent dimension used by the autoencoder encoder. |
| `--file_location` | `logs` | Directory to write the JSON log. (Must exist.) |
| `--file_name` | `experiment.json` | Log file name. |

### Outputs

#### 1) Encodings Cache

Encodings are cached under:

```
Encodings/
  dim_<latent_dim>/
    <list_path>_n<sample_size>_seed<seed>.npy
```

This caching is handled automatically by `_get_or_extract_features()` in `ShiftExperiment`.

#### 2) JSON Log

A JSON log is written to:

```
<file_location>/<file_name>
```

The log includes:

- calibration τ for each test
- sanity check results
- per-run target shift stats + detection decisions
- summary TPR for each statistical test

---

## Other Scripts

- `run_experiment.py`: older wrapper / experiment driver (kept for backward compatibility).
- `shift_experiment.py`: older CLI entrypoint referenced by the command generator.

If you are starting fresh, use **`experiment.py`**.
