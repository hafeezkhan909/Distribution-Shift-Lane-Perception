import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

import data_utils
from autoencoder import ConvAutoencoderFC
from mmd_test import mmd_test


# =========================================================
# Datasets
# =========================================================

class LaneImageDataset(Dataset):
    """Generic dataset for lane images given a root path and list file."""
    def __init__(self, root_dir, split="train", image_size=512):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        # list file logic
        if "Curvelanes" in root_dir:
            list_path = os.path.join(root_dir, split, f"{split}.txt")
        else:
            list_path = os.path.join(root_dir, "list", f"{split}.txt")

        if not os.path.exists(list_path):
            raise FileNotFoundError(f"List file not found: {list_path}")

        with open(list_path, "r") as f:
            self.image_paths = [line.strip() for line in f.readlines() if line.strip()]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx].lstrip("/")
        if "Curvelanes" in self.root_dir:
            img_path = os.path.join(self.root_dir, self.split, rel_path)
        else:
            img_path = os.path.join(self.root_dir, rel_path)

        img = Image.open(img_path).convert("RGB")
        return self.transform(img)


class LaneImageDatasetWithLabels(LaneImageDataset):
    """A Lane Image Dataset with Labels."""
    def __init__(self, root_dir, split="train", image_size=512):
        # Run the parent __init__
        super().__init__(root_dir, split, image_size)

        # Reparse the list file to avoid editing the parent class
        if "Curvelanes" in root_dir:
            list_path = os.path.join(root_dir, split, f"{split}.txt")
        else:
            list_path = os.path.join(root_dir, "list", f"{split}.txt")

        self.labels = []

        with open(list_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    parts = line.split()
                    # Assuming the order matches self.image_paths
                    self.labels.append(int(parts[1]))

        # Verify the consistency of length
        if len(self.image_paths) != len(self.labels):
            raise ValueError("Mismatch in number of images and labels")

    def __getitem__(self, index):
        """
        This method overrides the __getitem__ method of the LaneImageDataset class.
        """
        # Get path and label from self
        rel_path = self.image_paths[index].lstrip("/")
        label = self.labels[index]

        if "Curvelanes" in self.root_dir:
            img_path = os.path.join(self.root_dir, self.split, rel_path)
        else:
            img_path = os.path.join(self.root_dir, rel_path)

        img = Image.open(img_path).convert("RGB")

        # Apply the transform inherited from the parent
        return self.transform(img), label


# =========================================================
# Dataloader helpers
# =========================================================
def get_dataloader(dataset_name, split, batch_size, image_size, num_samples, block_idx=0):
    root = f"datasets/{dataset_name}"
    ds = LaneImageDataset(root, split, image_size)
    start, end = block_idx * num_samples, min((block_idx + 1) * num_samples, len(ds))
    subset = Subset(ds, list(range(start, end)))
    print(f"[INFO] {dataset_name} ({split}) → [{start}:{end}] ({len(subset)} samples)")
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def get_seeded_random_dataloader(dataset_name, split, batch_size, image_size, num_samples, seed):
    root = f"datasets/{dataset_name}"
    ds = LaneImageDataset(root, split, image_size)
    random.seed(seed)
    chosen = random.sample(range(len(ds)), min(num_samples, len(ds)))
    subset = Subset(ds, chosen)
    print(f"[INFO] {dataset_name} ({split}) → Random {len(chosen)} samples (seed={seed})")
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def get_dataloader_with_labels(dataset_name, split, batch_size, image_size, num_samples, block_idx=0):
    root = f"datasets/{dataset_name}"
    ds = LaneImageDatasetWithLabels(root, split, image_size)
    start, end = block_idx * num_samples, min((block_idx + 1) * num_samples, len(ds))
    subset = Subset(ds, list(range(start, end)))
    print(f"[INFO] {dataset_name} ({split}) → [{start}:{end}] ({len(subset)} samples)")
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def get_seeded_random_dataloader_with_labels(dataset_name, split, batch_size, image_size, num_samples, seed):
    root = f"datasets/{dataset_name}"
    ds = LaneImageDatasetWithLabels(root, split, image_size)
    random.seed(seed)
    chosen = random.sample(range(len(ds)), min(num_samples, len(ds)))
    subset = Subset(ds, chosen)
    print(f"[INFO] {dataset_name} ({split}) → Random {len(chosen)} samples (seed={seed})")
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def create_shifted_collate_function(shift_name, original_dimentions, _dataset):
    """
    A Function Facotry that creates a collate_fn for applying
    shifts. This collate_fn intercepts the batch of (tensor, label)
    tuples and converts them to Numpy, applies the shift, and
    converts them back.
    """
    def collate_with_shift(batch):
        # Unzip the batch of (tensor, label) tuples
        tensors, labels, *_ = zip(*batch)

        # Stacking the tensors
        stacked_tensors = torch.stack(tensors)

        # Converting the tensors to NumPy [N, H, W, C] in the [0, 255] range
        if stacked_tensors.dim() == 3:
            # Add a channel dimension at index 1
            # Shape becomes (BatchSize, 1, Height, Width)
            stacked_tensors = stacked_tensors.unsqueeze(1)
        
        numpy_tensor_batch = stacked_tensors.permute(0, 2, 3, 1).cpu().numpy()
        numpy_label_batch = (numpy_tensor_batch * 255.0).astype(np.uint8)

        # Shape for the shifting function
        original_shape = numpy_label_batch.shape
        numpy_tensor_batch_flattened = numpy_tensor_batch.reshape(original_shape[0], -1)  # (N, H*W*C)

        # Get the labels
        label_numpy_batch = np.array(labels)

        # Apply the shift using the imported data_utils func
        shifted_numpy_tensor_batch_flattened, shifted_labels = data_utils.apply_shift(
            X_te_orig=numpy_tensor_batch_flattened,
            y_te_orig=label_numpy_batch,
            shift=shift_name,
            orig_dims=original_dimentions,
            dataset=_dataset
        )

        # Reshape back to the image format
        shifted_numpy_tensor_batch = shifted_numpy_tensor_batch_flattened.reshape(original_shape)

        # Convert back to the Tensor Format [N, C, H, W] in [0.0, 1.0]
        shifted_numpy_tensor_batch = torch.from_numpy(shifted_numpy_tensor_batch).permute(0, 3, 1, 2)
        shifted_numpy_tensor_batch = shifted_numpy_tensor_batch.float() / 255.0

        # Return the images and labels (to match the default dataloader)
        shifted_label_tensor = torch.from_numpy(shifted_labels)
        return shifted_numpy_tensor_batch, shifted_label_tensor

    return collate_with_shift


def get_shifted_dataloader(dataset_name, split, batch_size, image_size, num_samples, seed, shift_name=None):
    """
    Creates the Dataloader that applies a specific shift using a collate function.
    """
    assert shift_name is not None
    root = f"datasets/{dataset_name}"
    ds = LaneImageDatasetWithLabels(root, split, image_size)
    random.seed(seed)
    chosen = random.sample(range(len(ds)), min(num_samples, len(ds)))
    subset = Subset(ds, chosen)

    # Create custom collate function
    collate_fn = create_shifted_collate_function(
        shift_name=shift_name,
        original_dimentions=[image_size, image_size, 3],
        _dataset=dataset_name
    )

    print(f"[INFO] {dataset_name} ({split} -> Random {len(chosen)} samples (seed={seed}) SHIFT: {shift_name})")

    # Pass the collate_fn to the Dataloader
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )


# =========================================================
# Feature extraction
# =========================================================
def extract_features(model, loader, device):
    model.eval()
    feats = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(device, non_blocking=True)
            z = model.encode(imgs)
            if z.dim() > 2:
                z = z.view(z.size(0), -1)
            feats.append(z.cpu().numpy())
    return np.concatenate(feats, axis=0)


def extract_features_with_labels(model, loader, device):
    model.eval()
    feats = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(device, non_blocking=True)
            z = model.encode(imgs)
            if z.dim() > 2:
                z = z.view(z.size(0), -1)
            feats.append(z.cpu().numpy())
    return np.concatenate(feats, axis=0)


# =========================================================
# Combined pipeline
# =========================================================
def main():
    # ----- CONFIG -----
    source = "Curvelanes"
    target = "Curvelanes"
    src_split = "train"
    tgt_split = "train"
    src_samples = 1000
    tgt_samples = 100
    block_idx = 0
    batch_size = 16
    image_size = 512
    num_calib = 100
    alpha = 0.05
    seed_base = 42
    # -------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("features", exist_ok=True)

    # ------------------ Model ------------------
    print("\nInitializing autoencoder...")
    model = ConvAutoencoderFC(latent_dim=512, pretrained=True).to(device)

    # ------------------ Source ------------------
    src_path = f"features/{source}_{src_split}_{src_samples}_{block_idx}.npy"
    if os.path.exists(src_path):
        src_feats = np.load(src_path)
        print(f"[INFO] Loaded source features → {src_path}")
    else:
        src_loader = get_dataloader(source, src_split, batch_size, image_size, src_samples, block_idx)
        src_feats = extract_features(model, src_loader, device)
        np.save(src_path, src_feats)
        print(f"[SAVED] {src_path} ({src_feats.shape})")

    # ------------------ Calibration ------------------
    print("\n[STEP] Calibration: same-domain (no shift)")
    null_stats = []
    for i in trange(num_calib, desc="Calibrating"):
        seed = seed_base + i
        tgt_loader = get_seeded_random_dataloader(source, src_split, batch_size, image_size, tgt_samples, seed)
        tgt_feats = extract_features(model, tgt_loader, device)
        t_stat, _ = mmd_test(src_feats, tgt_feats)
        null_stats.append(t_stat)

    null_stats = np.array(null_stats)
    tau = np.percentile(null_stats, 100 * (1 - alpha))
    print(f"\n[RESULT] τ({1 - alpha:.2f}) = {tau:.6f}")
    print(f"[RESULT] Mean MMD (no shift): {null_stats.mean():.6f} ± {null_stats.std():.6f}")
    np.save("features/calibration_null_mmd.npy", null_stats)

    # ------------------ Sanity Check ------------------
    print("\n[STEP] Sanity Check: CULane→CULane")
    seed_match = seed_base + 1
    tgt_loader = get_seeded_random_dataloader(target, tgt_split, batch_size, image_size, tgt_samples, seed_match)
    tgt_feats = extract_features(model, tgt_loader, device)
    mmd_val, _ = mmd_test(src_feats, tgt_feats)
    print(f"[CHECK] MMD(CULane→CULane) = {mmd_val:.6f}, τ = {tau:.6f}")
    print("✅ No shift detected (expected same-domain match)." if mmd_val <= tau else "❌ Unexpected shift.")

    # =========================================================
    # NEW SECTION: Cross-domain test (CULane → Curvelanes)
    # =========================================================
    # print("\n[STEP] Cross-domain test: CULane → Curvelanes using same τ")
    # target_cross = "CULane"

    # # ---- Run 100 random seeds ----
    # num_runs = 100
    # tpr_list = []
    # mmd_values = []

    # for run in trange(num_runs, desc="Cross-domain seeds"):
    #     seed_cross = seed_base + 100 + run  # avoid overlap with calibration seeds
    #     tgt_loader_cross = get_seeded_random_dataloader(
    #         target_cross, tgt_split, batch_size, image_size, tgt_samples, seed_cross
    #     )
    #     tgt_feats_cross = extract_features(model, tgt_loader_cross, device)
    #     mmd_cross, _ = mmd_test(src_feats, tgt_feats_cross)
    #     mmd_values.append(mmd_cross)
    #     detected = mmd_cross > tau
    #     tpr_list.append(int(detected))

    #     print(f"[RUN {run+1:03d}] MMD={mmd_cross:.6f} {'✅' if detected else '❌'}")

    # # ---- Summarize results ----
    # tpr = np.mean(tpr_list)
    # print("\n[RESULTS] Cross-domain detection summary")
    # print(f"Average MMD: {np.mean(mmd_values):.6f} ± {np.std(mmd_values):.6f}")
    # print(f"TPR (true positive rate) over {num_runs} runs: {tpr*100:.2f}%")
    # np.save("features/mmd_curvelanes_100runs.npy", np.array(mmd_values))
    # np.save("features/tpr_curvelanes_100runs.npy", np.array(tpr_list))

    # =======================================
    # Noise Tests (Gaussian, Knockout Shift, and Adversarial)
    # ========================================
    print(f"\n[STEP] Shifted-domain test: {source} -> {target} + shift")

    # Define all shifts to test from data_utils
    shifts_to_test = [
        'small_gn_shift_1.0'
        'medium_gn_shift_1.0',
        'large_gn_shift_1.0',
        'ko_shift_1.0',
        'ko_shift_0.5',
        'ko_shift_0.1',
        'small_img_shift_1.0',
        'medium_img_shift_1.0',
        'large_img_shift_1.0',
        'adversarial_shift_1.0',
        'medium_img_shift_0.5+ko_shift_0.1'
    ]

    # Number of runs per shift
    shift_runs = 50
    results = {}

    for shift_name in shifts_to_test:
        print(f"\n--- Testing shift: {shift_name} ---")
        tpr_list = []
        mmd_values = []

        for run in trange(shift_runs, desc=f"Shifted Domain Testing: {shift_name}"):
            # Use a new seed range
            seed_shift = seed_base + 1000 + run

            # Get shifted dataloader
            tgt_loader_shifted = get_shifted_dataloader(
                dataset_name=target,
                split=tgt_split,
                batch_size=batch_size,
                image_size=image_size,
                num_samples=tgt_samples,
                seed=seed_shift,
                shift_name=shift_name
            )

            # Extract the features from the shifted data
            try:
                tgt_feats_shifted = extract_features_with_labels(model, tgt_loader_shifted, device)
            except Exception as e:
                print(f"\n[ERROR] Failed to extract features for shift '{shift_name}'")
                print(f"\nError: {e}")
                print("This may be due to a missing helper function in data_utils")
                # Stop testing this shift
                break

            # Run the MMD test against the original source
            mmd_shifted, _ = mmd_test(src_feats, tgt_feats_shifted)
            mmd_values.append(mmd_shifted)

            detected = mmd_shifted > tau
            tpr_list.append(int(detected))

        # If the loop broke on the first try
        if not mmd_values:
            continue

        # ---- Summarize the results for the shift ----
        tpr = np.mean(tpr_list)
        results[shift_name] = {'tpr': tpr, 'mmd_mean': np.mean(mmd_values), 'mmd_std': np.std(mmd_values)}
        print(f"\n[RESULT: {shift_name}]")
        print(f"Average MMD: {np.mean(mmd_values):.6f} ± {np.std(mmd_values):.6f}")
        print(f"TPR over {shift_runs} runs: {tpr*100:.2f}%")

    # ----- Final Summary -----
    print(f"\n[FINAL RESULTS] Shift Detection summary")
    print("------------------------------------------------------------")
    print(f"{'ShiftName':<35} | {'TPR':<7} | {'Mean MMD':<15}")
    print("------------------------------------------------------------")
    for shift_name, result in results.items():
        print(f"{shift_name:<35} | {result['tpr']*100:6.2f}% | {result['mmd_mean']:.6f} ± {result['mmd_std']:.6f}")


if __name__ == "__main__":
    main()
