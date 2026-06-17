# Distribution-Shift-Lane-Perception

## Table of Contents

- [📝 Overview](#-overview)
- [⚡️ Setup](#-setup)
- [🚀 Quick Command Generator](#-command-generator-tool)
- [🔨 Manual `experiment.py` configuration](#-manual-experimentpy-configuration)
- [⚗️ Other Scripts (eg. training script)](#-other-scripts)

## 📝 Overview

The pipeline estimates an empirical threshold (`τ`) using *same-domain calibration*, and then tests whether samples from another dataset (or a perturbed version of the same dataset) come from a statistically different distribution.

The pipeline consists of:

1. **Extract Features:** Encodes features from the source and target data and generates statistics from the features.

![Extract Features](figures/readMeGraphics/a.svg)

2. **Calibration:** Forms a null distribution of test statistics by comparing a fixed source reference set against many randomly sampled sets from the *same source domain*. `τ` is set as the `(1 - α)` percentile of this null distribution.

![Calibration Diagram](figures/readMeGraphics/b.svg)

3. **Data Shift Test:** Compares the source reference features with samples from a target domain (e.g., CULane → Curvelanes) and reports the shift detection rate across runs.

![Data Shift Test Diagram](figures/readMeGraphics/c.svg)

## ⚡️ Setup

### Clone the Repository

```bash
git clone https://github.com/hafeezkhan909/Distribution-Shift-Lane-Perception.git
```

> Notice: Python 3.8 and above required

### Conda Environment Setup
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

### Pip Setup
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

| File Path in System | Path in `train.txt` |
| --- | --- |
| `./datasets/Curvelanes/split/001.jpg` | `split/001.jpg` |
| `./datasets/CULane/photoData/002.jpg` | `photoData/002.jpg` |

---

## 🚀 Command Generator Tool

> Click here to go to the command generation GUI: [https://suave101.github.io/Distribution-Shift-Lane-Perception-Command-Generator/](https://suave101.github.io/Distribution-Shift-Lane-Perception-Command-Generator/)
<a href="https://suave101.github.io/Distribution-Shift-Lane-Perception-Command-Generator/">
  <img width="1515" height="636" alt="Picture of Command Generator" src="https://github.com/user-attachments/assets/07c201e8-e561-44b7-bfa3-920d26c2c597" />
</a>

## 🔨 Manual `experiment.py` Configuration

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

## ⚗️ Other Scripts

### Autoencoder Training Script: `models/trainingScripts/train.py'

This script trains the **Configurable Autoencoder** used in Phase 2 and **automatically resumes** from the most recent checkpoint found in:

`checkpoints/Phase2/<dataset_name>/`

#### What it does
- Loads a dataset via `data.data_builder.get_dataloader(...)`
- Trains an autoencoder using **MSE reconstruction loss**
- Saves a checkpoint **every epoch**
- If checkpoints already exist for the dataset, it **loads the newest one** and continues training

#### Basic usage

```bash
python models/trainingScripts/train.py \
  --dataset_name <NAME> \
  --dataset_dir <PATH_TO_DATA_ROOT> \
  --dataset_list <PATH_TO_TRAIN_SPLIT_LIST>
```

#### Common options

- `--epochs` (default: `50`)  
- `--batch_size` (default: `128`)  
- `--image_size` (default: `512`)  
- `--learning_rate` (default: `1e-4`)  
- `--latent_dim` (default: `32`)  
- `--samples` (default: `100000`) limit number of samples used by the dataloader  
- `--cropImg` flag to crop the bottom half of images (if supported by the dataloader)  
- `--block_idx` selects which data block to read (dataset-dependent)  
- `--seed` for deterministic runs (default is 42)

#### Checkpoints (resume behavior)

Checkpoints are written to:

`checkpoints/Phase2/<dataset_name>/P2autoencoder_<dataset_name>_epoch_<N>.pth`

On startup, if any matching checkpoints exist in that folder, the script will:
1. pick the most recently created checkpoint file
2. load model + optimizer state
3. resume from the next epoch

#### Notes
- Runs on CUDA if available; otherwise falls back to CPU.
- If multiple GPUs are available, training uses `torch.nn.DataParallel`.

### List File Generator: `create_list_files.py`

#### List File Generator (`create_list_files.py`)

Use this utility to convert an experiment JSON log (containing `Image Paths` under `Individual Test Data`) into per-run list files plus a combined list. It writes **relative** paths by stripping the dataset root inferred from the experiment’s `source_list_path` / `target_list_path`.

**Process all JSON logs in a directory** (creates one output folder per JSON file):

```bash
python create_list_files.py --dir ./logs
```

Process a single JSON file (writes output under `--dir` if provided; otherwise alongside the JSON):
```bash
python create_list_files.py --file ./logs/experiment.json --dir ./logs
```

**Output structure** (example):

```
logs/
  experiment/
    <source_list_path>_to_<target_list_path>_1/
      Run_1.txt
      Run_2.txt
      ...
      full.txt
```

`Run_<id>.txt`: image paths for that run (one per line)
`full.txt`: de-duplicated union of all run paths for that experiment

---

If you are starting fresh, use **`experiment.py`** with either random or ImageNet weights.
