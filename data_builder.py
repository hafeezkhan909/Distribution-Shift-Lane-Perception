import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from data_utils import apply_shift

class LaneImageDataset(Dataset):
    """Generic dataset for lane images given a root path and list file."""

    def __init__(self, root_dir, split="train", image_size=512, dataShift=None, cropImage=False):
        self.shift = dataShift
        self.cropImage = cropImage
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        # list file logic: Needs modularization for any data loading.
        if "Curvelanes" in root_dir:
            list_path = os.path.join(root_dir, split, f"{split}.txt") # for Curvelanes txt file extraction
        else:
            list_path = os.path.join(root_dir, "list", f"{split}.txt") # for CULane txt file extraction 

        if not os.path.exists(list_path):
            raise FileNotFoundError(f"List file not found: {list_path}")

        with open(list_path, "r") as f:
            self.image_paths = [line.strip() for line in f.readlines() if line.strip()]

        self.transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx].lstrip("/")
        if "Curvelanes" in self.root_dir:
            img_path = os.path.join(self.root_dir, self.split, rel_path)
        else:
            img_path = os.path.join(self.root_dir, rel_path)

        img = Image.open(img_path).convert("RGB")
        if self.shift is not None:
            img = apply_shift(img, self.shift)
        if self.cropImage:
            w, h = img.size
            img = img.crop((0, h//2, w, h)) # left, top, right, bottom
            return self.transform(img)
        else:
            return self.transform(img)


# =========================================================
# Dataloader helpers
# =========================================================
def get_dataloader(
    dataset_name, split, batch_size, image_size, num_samples, block_idx=0, cropImage=False
):
    root = f"datasets/{dataset_name}"
    ds = LaneImageDataset(root, split, image_size, dataShift=None, cropImage=False)
    start, end = block_idx * num_samples, min((block_idx + 1) * num_samples, len(ds))
    subset = Subset(ds, list(range(start, end)))
    print(f"[INFO] {dataset_name} ({split}) → [{start}:{end}] ({len(subset)} samples)")
    return DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )


def get_seeded_random_dataloader(
    dataset_name, split, batch_size, image_size, num_samples, seed, shift=None, cropImage=False
):
    root = f"datasets/{dataset_name}"
    ds = LaneImageDataset(root, split, image_size, dataShift=shift, cropImage=False)
    random.seed(seed)
    chosen = random.sample(range(len(ds)), min(num_samples, len(ds)))
    subset = Subset(ds, chosen)
    # print(
    #     f"[INFO] {dataset_name} ({split}) → Random {len(chosen)} samples (seed={seed})"
    # )
    return DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
