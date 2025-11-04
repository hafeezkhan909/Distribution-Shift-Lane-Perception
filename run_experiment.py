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
# Dataset
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

        self.labels = []
        with open(list_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    parts = line.split()
                    self.image_paths.append(parts[0])
                    self.labels.append(int(parts[1]))


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
        label = self.labels[idx]
        return self.transform(img), label


# =========================================================
# Dataloader helpers
# =========================================================
def create_shifted_collate_fn(shift, original_dimensions, dataset):
    def collate_fn(batch):
        tensors, labels = zip(*batch)

        stack = torch.stack(batch)
        image_batch_numpy = stack.permute(0, 2, 3, 1).cpu().numpy()
        image_batch_numpy = (image_batch_numpy * 255.0).astype(np.uint8)

        original_shape = image_batch_numpy.shape
        image_flattened = np.reshape(original_shape[0], -1)

        label_batch_numpy = np.array(labels)

        image_flattened, label_batch_numpy = data_utils.apply_shift(

        )



def get_shifted_dataloader(dataset_name, split, batch_size, image_size, num_samples, block_idx=0):
    root = f"datasets/{dataset_name}"
    ds = LaneImageDataset(root, split, image_size)
    start, end = block_idx * num_samples, min((block_idx + 1) * num_samples, len(ds))
    subset = Subset(ds, list(range(start, end)))
    print(f"[INFO] {dataset_name} ({split}) → [{start}:{end}] ({len(subset)} samples)")

    collate_fn = create_shifted_collate_fn()

    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

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


# =========================================================
# Feature extraction
# =========================================================
def extract_features(model, loader, device):
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
    tgt_samples = 50
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
    print("\n[STEP] Cross-domain test: CULane → Curvelanes using same τ")
    target_cross = "CULane"

    # ---- Run 100 random seeds ----
    num_runs = 100
    tpr_list = []
    mmd_values = []

    for run in trange(num_runs, desc="Cross-domain seeds"):
        seed_cross = seed_base + 100 + run  # avoid overlap with calibration seeds
        tgt_loader_cross = get_seeded_random_dataloader(
            target_cross, tgt_split, batch_size, image_size, tgt_samples, seed_cross
        )
        tgt_feats_cross = extract_features(model, tgt_loader_cross, device)
        mmd_cross, _ = mmd_test(src_feats, tgt_feats_cross)
        mmd_values.append(mmd_cross)
        detected = mmd_cross > tau
        tpr_list.append(int(detected))

        print(f"[RUN {run+1:03d}] MMD={mmd_cross:.6f} {'✅' if detected else '❌'}")

    # ---- Summarize results ----
    tpr = np.mean(tpr_list)
    print("\n[RESULTS] Cross-domain detection summary")
    print(f"Average MMD: {np.mean(mmd_values):.6f} ± {np.std(mmd_values):.6f}")
    print(f"TPR (true positive rate) over {num_runs} runs: {tpr*100:.2f}%")
    np.save("features/mmd_curvelanes_100runs.npy", np.array(mmd_values))
    np.save("features/tpr_curvelanes_100runs.npy", np.array(tpr_list))

    # =========================================================
    # Noise tests (Gaussian, Knockout Shift, and Adversarial)
    # =========================================================


if __name__ == "__main__":
    main()
