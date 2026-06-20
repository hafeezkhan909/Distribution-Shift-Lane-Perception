# Distribution-Shift-Lane-Perception

## Method

The pipeline estimates a threshold `τ` calibrating on a source dataset, and then tests whether samples from another dataset come from a statistically different distribution using the `τ` threshold.

### 1. **Extract Source Features**

a. Selects a fixed set of images from the source dataset and encodes the images into features using a ResNet-18 autoencoder.

b. Randomly samples sets of images from the source dataset and encodes the images into features using the same ResNet-18 autoencoder.

c. Compares the fixed source feature set with the randomly sampled feature sets using statistical tests (Such as MMD, Energy, etc.).

![Extract Features](figures/readMeGraphics/a.svg)

2. **Source Calibration**

a. Sorts each set of statistical results in ascending order as a null distribution.

b. Determines the `τ` threshold by selecting the `(1 - α)`th percentile of the null distribution.

![Calibration Diagram](figures/readMeGraphics/b.svg)

3. **Data Shift Test** Compares the original fixed source feature set against many randomly sampled sets from the target dataset (e.g., Source: CULane → Target: Curvelanes) and reports the shift detection rate across runs.

a. Retrives the original fixed source feature set from the feature extraction step.

b. Randomly samples sets of images from the target dataset and encodes the images into features using the ResNet-18 autoencoder.

c. Compares the fixed source feature set with the randomly sampled feature sets using statistical tests (Such as MMD, Energy, etc.).

d. Determines if a shift is detected for each randomly sampled set of images from the target distribution by comparing the statistical test results to the `τ` threshold. If the statistical test value is larger than the `τ` threshold then a shift is detected.

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
│   ├── trainingScripts/                #
|   |   └── train.py                    # Training script for a custom Autoencoder
│   └── configurableAutoencoder.py      # 
├── utils/                              # Statistical tests and utilities
├── create_list_files.py                # 
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

Due to `experiment.py` having many arguments, we created the Command Generator Tool to make designing experiments quicker and easier!

Step 1: Generate Command with the [Command Generator GUI](https://suave101.github.io/Distribution-Shift-Lane-Perception-Command-Generator/)

Step 2: Copy Command

Step 3: Paste Command in Terminal

<a href="https://suave101.github.io/Distribution-Shift-Lane-Perception-Command-Generator/">
  <img width="1515" height="636" alt="Picture of Command Generator" src="https://github.com/user-attachments/assets/07c201e8-e561-44b7-bfa3-920d26c2c597" />
</a>

## `experiment.py` Configuration

`experiment.py` is the experiment runner.

### Basic Usage

#### Example (ImageNet weights)

```bash
python experiment.py \
  --source_dir ./datasets/CULane \
  --target_dir ./datasets/Curvelanes \
  --source_list_path ./datasets/CULane/list/train.txt \
  --target_list_path ./datasets/Curvelanes/list/train.txt \
  imagenet_weights

```

### Required Arguments

| Argument | Description |
| --- | --- |
| `--source_dir` | Root directory for the Source dataset. |
| `--target_dir` | Root directory for the Target dataset. |
| `--target_list_path` | Path to the text file containing target image paths. |

> `--source_list_path` defaults to `./datasets/CULane/list/train.txt`, but you will almost always want to set it explicitly.

### Optional Arguments

| Flag | Default | Description |
| --- | --- | --- |
| `--sample_size` | `1000` | Number of samples per run (source + each sampled target/calibration set). |
| `--num_runs` | `100` | Number of target runs used to compute detection rate. |
| `--batch_size` | `128` | Batch size for feature extraction. |
| `--image_size` | `512` | Resize images to `image_size x image_size`. |
| `--alpha` | `0.05` | Significance level; `τ` is set at the `(1-α)` percentile. |
| `--seed_base` | `42` | Base seed for all random sampling. |
| `--permutation_test_iterations` | `1000` | Number of permutations used for MMD/MMD_Agg/Energy p-values. Set to `0` to skip p-values. |
| `--latent_dim` | `32` | Latent dimension used by the autoencoder. |
| `--file_location` | `logs` | Directory to write the JSON log. **(Must exist)** |
| `--file_name` | `experiment.json` | Log file name. |

### Data Perturbation & Shift Arguments

Use these parameters to artificially induce domain shifts on the target dataset pipeline:

| Flag / Parameter | Default | Type | Description |
| --- | --- | --- | --- |
| `--gaussian_sigma` | `0.0` | `float` | Standard deviation of Gaussian noise added to target images. |
| `--crop_image` | `False` | `flag` | Include this flag to crop target images. |
| `--rotation_angle` | `0` | `float` | Rotation angle in degrees applied to target images. |
| `--width_shift_frac` | `0` | `float` | Fraction of total width to shift target images horizontally. |
| `--height_shift_frac` | `0` | `float` | Fraction of total height to shift target images vertically. |
| `--shear_angle` | `0` | `float` | Shear angle in degrees applied to target images. |
| `--zoom_factor` | `1.0` | `float` | Rescaling multiplier for zooming target images. |
| `--horizontal_flip` | `False` | `flag` | Include this flag to flip target images horizontally. |
| `--vertical_flip` | `False` | `flag` | Include this flag to flip target images vertically. |

### Results Logs

A JSON log is written to:

```
<file_location>/<file_name>

```

The log includes:

* calibration τ for each test
* per-run target shift stats + detection decisions
* summary TPR for each statistical test

### Modes: `custom_weights` vs `imagenet_weights vs `random_weights`

The weights you choose determine the behavior of the autoencoder. You can use the pre-trained `ImageNet1K_V1` autoencoder weights, randomly generated weights, or custom weights. The `custom_weights` mode for `experiment.py` allows you to use custom ResNet-18 weights as the autoencoder backbone. _See documentation for `models/trainingScripts/train.py` to learn more about generating your custom weights_

`experiment.py` has two subcommands:

* `imagenet_weights`: uses ImageNet weights (no custom weights directory required).
* `custom_weights`: uses a local weights directory for the ResNet-18 autoencoder. _Click here for more details on generating custom weights_

Mode-specific required args:

| Mode | Required Argument | Description |
| --- | --- | --- |
| `custom_weights` | `--model_weights_path` | Path to your custom ResNet-18 weights. |
| `imagenet_weights` | *(none)* | Uses ImageNet weights. |
| `random_weights` | *(none)* | Uses Randomly generated weights. |

This is the minimum required arguments to use custom weights:
```bash
python experiment.py \
  --source_dir ./datasets/CULane \
  --target_dir ./datasets/Curvelanes \
  --source_list_path ./datasets/CULane/list/train.txt \
  --target_list_path ./datasets/Curvelanes/list/train.txt \
  custom_weights \
  --model_weights_path ./weights/ASSIST_TAXI.pth
```

## Other Scripts

### Autoencoder Training Script: `models/trainingScripts/train.py`

This script trains the **Configurable Autoencoder** used in Phase 2 and **automatically resumes** from the most recent checkpoint found in:

`checkpoints/Phase2/<dataset_name>/`

#### What it does
- Loads a dataset
- Trains an autoencoder using **MSE reconstruction loss**
- Saves a checkpoint **every epoch**

#### Basic usage

```bash
python models/trainingScripts/train.py \
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
