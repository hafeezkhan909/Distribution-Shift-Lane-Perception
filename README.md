# Distribution-Shift-Lane-Perception

## Method

The pipeline estimates a threshold `τ` calibrating on a source dataset, and then tests whether samples from another dataset come from a statistically different distribution using the `τ` threshold.

### 1. **Extract Source Features**

a. Selects a fixed set of images as the **_Source Set_** of images from _**Dataset A**_. Then the pipeline randomly samples sets of images from _**Dataset A**_ as _**Target Sets**_ to compare with the _**Source Set**_ later.

b. Reduces the dimensions of the **_Source Set_** images and the _**Target Sets**_ images using a ResNet-18 encoder using ImageNet1K_V1, Randomly Generated, or Custom Weights. This generates a _**Source Feature Set**_ and _**Target Feature Sets**_.

c. Compares the **_Source Feature Set_** with the _**Target Feature Sets**_ using a two sample statistical tests (Such as MMD, Energy, BKS, etc.). These tests quantify the distributional distance between the feature sets within their high-dimensional embedding space.

![Extract Features](figures/readMeGraphics/a.svg)

### 2. **Source Calibration**

a. Sorts each set of statistical two sample test results (Such as MMD, etc.) in ascending order to act as a set of null distributions.

b. Determines the `τ` threshold for each two sample test by selecting the `(1 - α)`th percentile of each of the null distributions.

![Calibration Diagram](figures/readMeGraphics/b.svg)

### 3. **Data Shift Test**

a. Retrives the original _**Source Set**_ of images from the first step. Then randomly samples sets of images from _**Dataset B**_ that will now act as the new _**Target Set**_.

b. Encodes the images from both the _**Source Set**_ and _**Target Set**_ into features using the ResNet-18 encoder using the same weights from the first step. This generates a _**Source Feature Set**_ (_which is the exact same as the Source Feature Set from the first step_) and new _**Target Feature Sets**_.

c. Compares the **_Source Feature Set_** with the _**Target Feature Sets**_ using a two sample statistical tests (Such as MMD, Energy, BKS, etc.).

d. Determines if a shift is detected for each set of _**Target Features**_ by comparing the two sample test results with the `τ` threshold. If the statistical test value is larger than the `τ` threshold then a shift is detected.

![Data Shift Test Diagram](figures/readMeGraphics/c.svg)

## Code Structure

```plaintext
.
├── data/                               # Data processing and logging scripts
│   ├── data_builder.py                 # Creates torch dataloaders
│   ├── data_logging.py                 # Generates experiment JSON Results
│   └── data_utils.py                   # Data Perturbation tools (eg. Flip Vertically)
├── figures/                            # Directory for generated figures
├── models/                             # Model architectures and training
│   ├── train.py                        # Training script for a custom Autoencoder
│   └── configurableAutoencoder.py      # ResNet-18 Autoencoder used for encoding of images
├── utils/                              # Statistical tests and utilities
├── create_list_files.py                # Tool that generates image dataset list files
├── environment.yml                     # Conda environment configuration
└── experiment.py                       # Main experiment runner
```

## Data Preperation

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

## Setup

### Clone the Repository

```bash
git clone https://github.com/hafeezkhan909/Distribution-Shift-Lane-Perception.git
```

> Notice: Python 3.8 and above required

### Conda Environment Setup

#### 1. Install everything using the environment.yml file
```
conda env create -f environment.yml
```

#### 2. Activate the environment
```
conda activate distribution_shift_perception
```

#### 3. Install the torch_two_sample dependency
```
git clone https://github.com/josipd/torch-two-sample.git
```
```
cd torch-two-sample/
```
```
python setup.py install
```

### Pip Setup
#### 1. Install dependencies
```
pip install scipy torch torchvision tqdm Pillow
```
```
pip install "numpy<2.0"
```

#### 2. Install the torch_two_sample dependency
```
git clone https://github.com/josipd/torch-two-sample.git
```
```
cd torch-two-sample/
```
```
python setup.py install
```

## 🚀 Command Generator Tool

> Click here to go to the command generation GUI: [https://suave101.github.io/Distribution-Shift-Lane-Perception-Command-Generator/](https://suave101.github.io/Distribution-Shift-Lane-Perception-Command-Generator/)

Here is the link to the tutorial video on how to use the Command Generator Tool:

[![Lane Perception Command Generator Tutorial](https://img.youtube.com/vi/ovT6UFNmc84/0.jpg)](https://youtu.be/ovT6UFNmc84)

Due to `experiment.py` having many arguments, we created the Command Generator Tool to make designing experiments quicker and easier!

Step 1: Generate Command with the [Command Generator GUI](https://suave101.github.io/Distribution-Shift-Lane-Perception-Command-Generator/)

Step 2: Copy Command

Step 3: Paste Command in Terminal

<a href="https://suave101.github.io/Distribution-Shift-Lane-Perception-Command-Generator/">
  <img width="1515" height="636" alt="Picture of Command Generator" src="https://github.com/user-attachments/assets/07c201e8-e561-44b7-bfa3-920d26c2c597" />
</a>

## `experiment.py` Configuration

`experiment.py` is the main experiment runner.

### Quick Start Examples

#### 1. Minimal Example (ImageNet Weights)

```bash
python experiment.py \
  --source_dir ./datasets/CULane \
  --target_dir ./datasets/Curvelanes \
  --source_list_path ./datasets/CULane/list/train.txt \
  --target_list_path ./datasets/Curvelanes/list/train.txt \
  imagenet_weights

```

#### 2. Minimal Example (Custom Weights)

```bash
python experiment.py \
  --source_dir ./datasets/CULane \
  --target_dir ./datasets/Curvelanes \
  --source_list_path ./datasets/CULane/list/train.txt \
  --target_list_path ./datasets/Curvelanes/list/train.txt \
  custom_weights \
  --model_weights_path ./weights/ASSIST_TAXI.pth

```

#### 3. Advanced Example (Using Optional & Perturbation Arguments)

To modify hyperparameters and artificially induce a domain shift on your target dataset, append the optional arguments **before** or **after** the subcommand as needed:

```bash
python experiment.py \
  --source_dir ./datasets/CULane \
  --target_dir ./datasets/Curvelanes \
  --sample_size 500 \
  --batch_size 64 \
  --gaussian_sigma 1.5 \
  --crop_image \
  --horizontal_flip \
  imagenet_weights

```

### Core Arguments Reference

#### Required Configuration

| Argument | Description |
| --- | --- |
| `--source_dir` | Root directory for the Source dataset. |
| `--target_dir` | Root directory for the Target dataset. |
| `--target_list_path` | Path to the text file containing target image paths. |

> `--source_list_path` defaults to `./datasets/CULane/list/train.txt`, but you will almost always want to set it explicitly.

#### Optional Execution Hyperparameters

These tweak the runner's behavior, logging, and statistical thresholds.

| Flag | Default | Type | Description |
| --- | --- | --- | --- |
| `--sample_size` | `1000` | `int` | Number of samples per run (source + each sampled target/calibration set). |
| `--num_runs` | `100` | `int` | Number of target runs used to compute detection rate. |
| `--batch_size` | `128` | `int` | Batch size for feature extraction. |
| `--image_size` | `512` | `int` | Resize images to `image_size x image_size`. |
| `--alpha` | `0.05` | `float` | Significance level; $\tau$ is set at the $(1-\alpha)$ percentile. |
| `--seed_base` | `42` | `int` | Base seed for all random sampling. |
| `--permutation_test_iterations` | `1000` | `int` | Number of permutations for MMD/Energy p-values. Set to `0` to skip. |
| `--latent_dim` | `32` | `int` | Latent dimensions the ResNet-18 encoder ends with. |
| `--file_location` | `logs` | `str` | Directory to write the JSON log. **(Must exist)** |
| `--file_name` | `experiment.json` | `str` | Log file name. |

#### Data Perturbation & Shift Arguments

Use these parameters to artificially induce domain shifts on the target dataset pipeline.

* **Valued Arguments** require a number after them (e.g., `--rotation_angle 45`).
* **Flags** just need to be named to activate (e.g., `--crop_image`).

| Flag / Parameter | Default | Type | Description |
| --- | --- | --- | --- |
| `--gaussian_sigma` | `0.0` | `float` | Standard deviation of Gaussian noise added to target images. |
| `--rotation_angle` | `0` | `float` | Rotation angle in degrees applied to target images. |
| `--width_shift_frac` | `0` | `float` | Fraction of total width to shift target images horizontally. |
| `--height_shift_frac` | `0` | `float` | Fraction of total height to shift target images vertically. |
| `--shear_angle` | `0` | `float` | Shear angle in degrees applied to target images. |
| `--zoom_factor` | `1.0` | `float` | Rescaling multiplier for zooming target images. |
| `--crop_image` | *Disabled* | **Flag** | Include this flag to crop target images. |
| `--horizontal_flip` | *Disabled* | **Flag** | Include this flag to flip target images horizontally. |
| `--vertical_flip` | *Disabled* | **Flag** | Include this flag to flip target images vertically. |

### Weight Modes: `custom_weights` vs `imagenet_weights` vs `random_weights`

The backend encoder behavior is determined by the subcommand you append to the end of your execution string.

| Mode Subcommand | Required Extra Arguments | Description |
| --- | --- | --- |
| `imagenet_weights` | *(none)* | Uses pre-trained `ImageNet1K_V1` weights. |
| `random_weights` | *(none)* | Uses randomly initialized weights. |
| `custom_weights` | `--model_weights_path` | Uses custom ResNet-18 weights (e.g., from `train.py`). |

---

### Results Logs

A JSON log is written to `<file_location>/<file_name>` containing:

* Calibration $\tau$ for each test.
* Per-run target shift stats + detection decisions.
* Summary TPR (True Positive Rate) for each statistical test.

## Other Scripts

### Autoencoder Training Script: `models/train.py`

This script trains the **Encoder** used in Phase 2 and **automatically resumes** from the most recent checkpoint found in:

`checkpoints/Phase2/<dataset_name>/`

#### What it does
- Loads a dataset
- Trains an autoencoder using **MSE reconstruction loss**
- Saves a checkpoint **every epoch**

#### Basic usage

```bash
python models/train.py \
  --dataset_name <NAME> \
  --dataset_dir <PATH_TO_DATA_ROOT> \
  --dataset_list <PATH_TO_TRAIN_SPLIT_LIST>
```

#### Optional Args

> Note: Ensure that when you run `experiment.py` the `latent_dim` arg matches the one you use in `train.py`

- `--epochs` (default: `50`)  
- `--batch_size` (default: `128`)  
- `--image_size` (default: `512`)  
- `--learning_rate` (default: `1e-4`)  
- `--latent_dim` (default: `32`) 
- `--samples` (default: `100000`) limit number of samples used by the dataloader  
- `--cropImg` flag to crop the bottom half of images (if supported by the dataloader)  
- `--block_idx` selects which data block to read (dataset-dependent)  
- `--seed` (default is 42) for deterministic runs

#### Checkpoints (resume behavior)

Checkpoints are written to:

`checkpoints/Phase2/<dataset_name>/P2autoencoder_<dataset_name>_epoch_<N>.pth`

On startup, if any matching checkpoints exist in that folder, the script will:
1. pick the most recently created checkpoint file
2. load model + optimizer state
3. resume from the next epoch

### List File Generator: `create_list_files.py`

Use this utility to convert an experiment JSON log (containing `Image Paths` under `Individual Test Data`) into list files. It writes **relative** paths by stripping the dataset root from the experiment’s `source_list_path` / `target_list_path`.

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
