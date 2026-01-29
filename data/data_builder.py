import os
import random
import warnings
from typing import Any, List, Optional

from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
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

    def get_image_path(self, idx: int) -> str:
        """Resolves the full file path for a given index.

        Child classes can override this method if their path logic differs.
        """
        # Normalize the listed path and handle absolute paths
        listed = self.image_paths[idx]
        # If the list file contains an absolute path and that path actually
        # exists on this filesystem, return it. Some list files start paths
        # with a leading slash (e.g. '/driver_23_30frame/...') which are not
        # true absolute paths for this machine — treat those as relative to
        # `root_dir` if the absolute form does not exist.
        if os.path.isabs(listed):
            if os.path.exists(listed):
                return listed
            # otherwise treat as relative by stripping leading slashes
        rel = listed.lstrip("/")

        # Candidate locations to try (ordered by likelihood)
        candidates = [
            os.path.join(self.root_dir, rel),
            os.path.join(self.root_dir, "images", rel),
            os.path.join(self.root_dir, "train", rel),
            os.path.join(self.root_dir, "test", rel),
            os.path.join(self.root_dir, "valid", rel),
            os.path.join(self.root_dir, "list", rel),
            os.path.join(self.root_dir, "laneseg_label_w16", rel),
            os.path.join(self.root_dir, "laneseg_label_w16_test", rel),
        ]

        for c in candidates:
            if os.path.exists(c):
                return c

        # Fallback: return the primary join even if it doesn't exist (caller will raise on open)
        return candidates[0]

    def __getitem__(self, idx: int) -> Any:
        """Retrieves the image at the specified index and applies all steps.

        Steps include: path construction, image loading, optional shift,
        optional crop, resizing, and conversion to tensor.

        Args:
            idx: The index of the image path to retrieve.

        Returns:
            The transformed image data, typically a torch.Tensor (C, H, W).
        """

        img_path: str = self.get_image_path(idx)

        # 2. Load the image
        try:
            img: Image.Image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError) as e:
            print(f"Error loading image: {img_path}")
            raise e

        # 3. Apply optional shift
        if self.shift is not None:
            img = apply_shift(img, self.shift)

        # 4. Apply optional crop to the bottom half
        if self.cropImg:
            w, h = img.size
            # Crop region: (left, top, right, bottom)
            img = img.crop((0, h // 2, w, h))

        # 5. Apply the standard transformation pipeline
        return self.transform(img)


class LaneImageDataset(Dataset):
    """Generic dataset for lane images given a root path and list file."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 512,
        cropImg: bool = False,
        dataShift: DataShift = None,
    ):
        warnings.warn(
            "LaneImageDataset is pending deprecation. Use ImageDataset directly.",
            PendingDeprecationWarning,
            stacklevel=2,
        )

        # Determine list_path based on dataset name
        if "Curvelanes" in root_dir:
            list_path = os.path.join(root_dir, split, f"{split}.txt")
            self._is_curvelanes = True
        else:
            list_path = os.path.join(root_dir, "list", f"{split}.txt")
            self._is_curvelanes = False

        self.split = split

        # Initialize parent class
        super().__init__(
            root_dir=root_dir,
            list_path=list_path,
            image_size=image_size,
            cropImg=cropImg,
            dataShift=dataShift,
        )

    def get_image_path(self, idx: int) -> str:
        """Overrides the parent path logic to handle dataset specific structures."""
        rel_path = self.image_paths[idx].lstrip("/")

        if self._is_curvelanes:
            return os.path.join(self.root_dir, self.split, rel_path)
        else:
            return os.path.join(self.root_dir, rel_path)


# --- Dataloader Helpers ---


def get_dataloader(
    root_dir: str,
    list_path: str,
    batch_size: int,
    image_size: int,
    num_samples: int,
    cropImg: bool,
    block_idx: int = 0,
):

    # Instantiate the specific dataset class
    ds = ImageDataset(
        root_dir=root_dir, list_path=list_path, image_size=image_size, cropImg=cropImg
    )

    start = block_idx * num_samples
    end = min((block_idx + 1) * num_samples, len(ds))

    # Validate indices
    if start >= len(ds):
        raise ValueError(
            f"Block index {block_idx} is out of range for dataset size {len(ds)}"
        )

    # Generate the list of indices for this block
    indices = list(range(start, end))

    # Extract the full paths for these specific indices using the class method
    image_paths = [ds.get_image_path(i) for i in indices]

    subset = Subset(ds, indices)
    print(f"[INFO] ({root_dir}) → [{start}:{end}] ({len(subset)} samples)")

    return [
        DataLoader(
            subset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True, persistent_workers=True
        ),
        image_paths,
    ]


def get_seeded_random_dataloader(
    root_dir: str,
    list_path: str,
    batch_size: int,
    image_size: int,
    num_samples: int,
    seed: int,
    cropImg: bool,
    shift: Optional[DataShift],
):

    ds = ImageDataset(
        root_dir=root_dir,
        list_path=list_path,
        image_size=image_size,
        cropImg=cropImg,
        dataShift=shift,
    )

    # Set the random seed for reproducibility
    random.seed(seed)

    # Randomly sample indices without replacement
    chosen_indices = random.sample(range(len(ds)), min(num_samples, len(ds)))

    # Extract the full paths for these specific indices using the class method
    image_paths = [ds.get_image_path(i) for i in chosen_indices]

    # Generate the subset
    subset = Subset(ds, chosen_indices)

    return [
        DataLoader(
            subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        ),
        image_paths,
    ]


def get_concat_dataloader(
    root_dirs: List[str],
    list_paths: List[str],
    batch_sizes: List[int],
    image_sizes: List[int],
    num_samples: List[int],
    cropImg: List[bool],
    block_idx: List[int],
    seeds: List[int]
):
    """Creates a single combined dataloader from multiple datasets.
    
    Returns:
        A list containing:
            - A single DataLoader with all datasets combined
            - List of all image paths across datasets
    """
    # Ensure all input lists have the same length
    assert len(root_dirs) == len(list_paths) == len(batch_sizes) == len(
        image_sizes
    ) == len(num_samples) == len(cropImg) == len(block_idx), "All input lists must have the same length."

    subsets = []
    all_image_paths = []

    print("[INFO] Mixed Dataloader Configuration:")

    # Instantiate the specific dataset classes
    for i in range(len(root_dirs)):
        ds = ImageDataset(
            root_dir=root_dirs[i],
            list_path=list_paths[i],
            image_size=image_sizes[i],
            cropImg=cropImg[i],
        )

        start = block_idx[i] * num_samples[i]
        end = min((block_idx[i] + 1) * num_samples[i], len(ds))

        # Validate indices
        if start >= len(ds):
            raise ValueError(
                f"Block index {block_idx[i]} is out of range for dataset size {len(ds)}"
            )

        # Set seed unique to this dataset index for reproducibility
        random.seed(seeds[i])

        # Randomly sample indices without replacement
        chosen_indices = random.sample(range(len(ds)), min(num_samples[i], len(ds)))

        # Extract the full paths for these specific indices
        image_paths = [ds.get_image_path(j) for j in chosen_indices]
        all_image_paths.extend(image_paths)  # Flatten into single list

        subset = Subset(ds, chosen_indices)
        subsets.append(subset)
        print(f"[INFO] ({root_dirs[i]}) → [{start}:{end}] ({len(subset)} samples)")

    # Concatenate all subsets into one dataset
    combined_dataset = ConcatDataset(subsets)
    
    print(f"[INFO] Total combined samples: {len(combined_dataset)}")
    
    # Use the first batch_size (or make batch_size a single int parameter)
    batch_size = batch_sizes[0] if isinstance(batch_sizes, list) else batch_sizes
    
    combined_dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True
    )

    return [combined_dataloader, all_image_paths]
