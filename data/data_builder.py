import os
import random
from typing import Any, List

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from data.data_utils import DataShift, apply_shift


class ImageDataset(Dataset):
    """A generic PyTorch dataset for loading images from a file list.

    This class loads image paths from a specified list file and provides standard
    image transformations (resizing, optional cropping, optional shifting).

    Attributes:
        shift: The data shift parameter used for augmentation/correction.
        cropImg: Flag indicating whether to crop the image to the bottom half.
        root_dir: The base path of the dataset where images are located.
        list_path: The full path to the text file containing relative image paths.
        image_size: The target image size for transformation.
        image_paths: A list of relative paths to all images in the dataset.
        transform: The torchvision transformation pipeline.
    """

    def __init__(
        self,
        root_dir: str,
        list_path: str,
        image_size: int = 512,
        cropImg: bool = False,
        dataShift: DataShift = None,
    ) -> None:
        """Initializes the ImageDataset.

        Args:
            root_dir: The root directory where the dataset images are located.
            list_path: The full file path to the list file (e.g., 'train.txt')
                containing relative paths to images, one per line.
            image_size: The target size (width and height) for resizing the
                images. Defaults to 512.
            cropImg: If True, crops the image to the bottom half (0, h//2, w, h)
                after loading but before final transformation. Defaults to False.
            dataShift: Optional parameter used by the external `apply_shift`
                function. Defaults to None.

        Raises:
            FileNotFoundError: If the list file specified by `list_path` is not found.
        """
        self.shift: DataShift = dataShift
        self.cropImg: bool = cropImg
        self.root_dir: str = root_dir
        self.image_size: int = image_size

        if not os.path.exists(list_path):
            raise FileNotFoundError(f"List file not found: {list_path}")

        # Load image paths from the list file
        with open(list_path, "r") as f:
            self.image_paths: List[str] = [
                line.strip() for line in f.readlines() if line.strip()
            ]

        # Define the common transformation pipeline
        self.transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        """Returns the total number of images in the dataset.

        Returns:
            The number of image file paths loaded.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Any:
        """Retrieves the image at the specified index and applies all steps.

        Steps include: path construction, image loading, optional shift,
        optional crop, resizing, and conversion to tensor.

        Args:
            idx: The index of the image path to retrieve.

        Returns:
            The transformed image data, typically a torch.Tensor (C, H, W).
        """

        img_path: str = os.path.join(self.root_dir, self.image_paths[idx].lstrip("/"))

        # Load the image
        img: Image.Image = Image.open(img_path).convert("RGB")

        # Apply optional shift
        if self.shift is not None:
            img = apply_shift(img, self.shift)

        # Apply optional crop to the bottom half
        if self.cropImg:
            w, h = img.size
            # Crop region: (left, top, right, bottom)
            img = img.crop((0, h // 2, w, h))

        # Apply the standard transformation pipeline
        return self.transform(img)


@PendingDeprecationWarning
class LaneImageDataset(Dataset):
    """Generic dataset for lane images given a root path and list file."""

    def __init__(
        self, root_dir, split="train", image_size=512, cropImg=False, dataShift=None
    ):
        self.shift = dataShift
        self.cropImg = cropImg
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        # list file logic: Needs modularization for any data loading.
        if "Curvelanes" in root_dir:
            list_path = os.path.join(
                root_dir, split, f"{split}.txt"
            )  # for Curvelanes txt file extraction
        else:
            list_path = os.path.join(
                root_dir, "list", f"{split}.txt"
            )  # for CULane txt file extraction

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
        if self.cropImg:
            w, h = img.size
            img = img.crop((0, h // 2, w, h))  # left, top, right, bottom
            return self.transform(img)
        else:
            return self.transform(img)


# Dataloader helpers
def get_dataloader(
    dataset_name, split, batch_size, image_size, num_samples, cropImg, block_idx=0
):
    root = f"datasets/{dataset_name}"
    ds = LaneImageDataset(root, split, image_size, cropImg)
    start, end = block_idx * num_samples, min((block_idx + 1) * num_samples, len(ds))
    subset = Subset(ds, list(range(start, end)))
    print(f"[INFO] {dataset_name} ({split}) → [{start}:{end}] ({len(subset)} samples)")
    return DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )


def get_seeded_random_dataloader(
    dataset_name, split, batch_size, image_size, num_samples, seed, cropImg, shift
):
    root = f"datasets/{dataset_name}"
    ds = LaneImageDataset(root, split, image_size, cropImg, dataShift=shift)
    random.seed(seed)
    chosen = random.sample(range(len(ds)), min(num_samples, len(ds)))
    subset = Subset(ds, chosen)
    # print(
    #     f"[INFO] {dataset_name} ({split}) → Random {len(chosen)} samples (seed={seed})"
    # )
    return DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
