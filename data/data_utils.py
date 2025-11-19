# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import numpy as np
from enum import Enum
from PIL import Image
from torchvision.transforms import functional as F
import os
from abc import ABC, abstractmethod


# -------------------------------------------------
# SHIFT TYPES
# -------------------------------------------------
class ShiftTypes(Enum):
    """Enumeration of the available deterministic data shift types."""

    GAUSSIAN = "Gaussian"
    ROTATION = "Rotation"
    TRANSLATION = "Translation"
    SHEAR = "Shear"
    ZOOM = "Zoom"
    FLIP_HORIZ = "Flip Horizontally"
    FLIP_VERT = "Flip Vertically"


# =========================================================
# Data Shift Definitions
# =========================================================
class DataShift(ABC):
    """Abstract base class for a data shift operation."""

    def __init__(self):
        self.type = None

    @abstractmethod
    def __str__(self):
        pass


class GaussianShift(DataShift):
    """Defines a data shift by adding Gaussian noise.

    Attributes:
        std (float): The standard deviation of the Gaussian distribution.
        mean (float): The mean of the Gaussian distribution.
    """

    def __init__(self, std: float = 0, mean: float = 0):
        self.type = ShiftTypes.GAUSSIAN
        self.std = std
        self.mean = mean

    def __str__(self):
        return (
            f"DataShift: (Type: {self.type.value}, Std: {self.std}, Mean: {self.mean})"
        )


class RotationShift(DataShift):
    """Defines a deterministic rotation.

    Attributes:
        angle (float): The angle in degrees to rotate the image.
    """

    def __init__(self, angle: float = 0):
        self.type = ShiftTypes.ROTATION
        self.angle = angle

    def __str__(self):
        return f"DataShift: (Type: {self.type.value}, Angle: {self.angle})"


class TranslationShift(DataShift):
    """
    Defines a deterministic horizontal and vertical shift based on fractions
    of the image dimensions.

    Attributes:
        width_shift_frac (float): Fraction of total width to shift.
                                  Positive values shift right, negative shift left.
        height_shift_frac (float): Fraction of total height to shift.
                                   Positive values shift down, negative shift up.
    """

    def __init__(self, width_shift_frac: float = 0, height_shift_frac: float = 0):
        self.type = ShiftTypes.TRANSLATION
        self.height_shift_frac = height_shift_frac
        self.width_shift_frac = width_shift_frac

    def __str__(self):
        return f"DataShift: (Type: {self.type.value}, H-Shift: {self.height_shift_frac}, W-Shift: {self.width_shift_frac})"


class ShearShift(DataShift):
    """
    Defines a deterministic shear by a specific angle.

    Attributes:
        angle (float): The angle in degrees to shear the image.
    """

    def __init__(self, angle: float = 0):
        self.type = ShiftTypes.SHEAR
        self.angle = angle

    def __str__(self):
        return f"DataShift: (Type: {self.type.value}, Angle: {self.angle})"


class ZoomShift(DataShift):
    """
    Defines a deterministic zoom by a specific factor.

    Attributes:
        zoom_factor (float): The scaling factor. 1.0 is no zoom.
                             > 1.0 zooms in, < 1.0 zooms out.
    """

    def __init__(self, zoom_factor: float = 1.0):
        self.type = ShiftTypes.ZOOM
        self.zoom_factor = zoom_factor

    def __str__(self):
        return f"DataShift: (Type: {self.type.value}, Factor: {self.zoom_factor})"


class HorizontalFlipShift(DataShift):
    """Defines a deterministic horizontal flip."""

    def __init__(self):
        self.type = ShiftTypes.FLIP_HORIZ

    def __str__(self):
        return f"DataShift: (Type: {self.type.value})"


class VerticalFlipShift(DataShift):
    """Defines a deterministic vertical flip."""

    def __init__(self):
        self.type = ShiftTypes.FLIP_VERT

    def __str__(self):
        return f"DataShift: (Type: {self.type.value})"


# -------------------------------------------------
# SHIFT APPLICATOR
# -------------------------------------------------


def apply_shift(image: Image.Image, dataShift: DataShift) -> Image.Image:
    """
    Dispatcher function that applies a specific DataShift to a PIL image.

    Args:
        image (Image.Image): The input PIL image.
        dataShift (DataShift): An object (e.g., GaussianShift, RotationShift)
                                defining the shift to apply.

    Returns:
        Image.Image: The transformed PIL image.
    """
    # print(f"Applying: {dataShift}")

    if dataShift.type == ShiftTypes.GAUSSIAN:
        return add_gaussian_noise(image, mean=dataShift.mean, std=dataShift.std)

    elif dataShift.type == ShiftTypes.ROTATION:
        return apply_rotation(image, angle=dataShift.angle)

    elif dataShift.type == ShiftTypes.TRANSLATION:
        return apply_translation(
            image,
            width_shift_frac=dataShift.width_shift_frac,
            height_shift_frac=dataShift.height_shift_frac,
        )

    elif dataShift.type == ShiftTypes.SHEAR:
        return apply_shear(image, angle=dataShift.angle)

    elif dataShift.type == ShiftTypes.ZOOM:
        return apply_zoom(image, zoom_factor=dataShift.zoom_factor)

    elif dataShift.type == ShiftTypes.FLIP_HORIZ:
        return apply_horizontal_flip(image)

    elif dataShift.type == ShiftTypes.FLIP_VERT:
        return apply_vertical_flip(image)

    else:
        print(
            f"Warning: Shift type '{dataShift.type}' not recognized. Returning original image."
        )
        return image


# -------------------------------------------------
# DATA UTILS (All Deterministic)
# -------------------------------------------------


# --- 1. Rotation (Angle) ---
def apply_rotation(pil_img: Image.Image, angle: float) -> Image.Image:
    """
    Applies a specific rotation by a given angle in degrees.

    Args:
        pil_img (Image.Image): The input PIL image.
        angle (float): The angle in degrees to rotate.

    Returns:
        Image.Image: The rotated image.
    """
    if angle == 0.0:
        return pil_img
    return F.rotate(pil_img, angle=angle, fill=0)  # fill=0 sets the background to black


# --- 2. Translation (Width/Height Shift) ---
def apply_translation(
    pil_img: Image.Image, width_shift_frac: float, height_shift_frac: float
) -> Image.Image:
    """
    Applies a deterministic horizontal and vertical shift based on fractions.

    Args:
        pil_img (Image.Image): The input PIL image.
        width_shift_frac (float): Fraction of total width to shift.
                                  Positive values shift right, negative left.
        height_shift_frac (float): Fraction of total height to shift.
                                   Positive values shift down, negative up.

    Returns:
        Image.Image: The translated image.
    """
    if width_shift_frac == 0.0 and height_shift_frac == 0.0:
        return pil_img
    width, height = pil_img.size
    tx = int(width * width_shift_frac)
    ty = int(height * height_shift_frac)
    return F.affine(
        pil_img, angle=0.0, translate=[tx, ty], scale=1.0, shear=[0.0, 0.0], fill=0
    )


# --- 3. Shear ---
def apply_shear(pil_img: Image.Image, angle: float) -> Image.Image:
    """
    Applies a deterministic shear by a specific angle in degrees.

    Args:
        pil_img (Image.Image): The input PIL image.
        angle (float): The angle in degrees to shear.

    Returns:
        Image.Image: The sheared image.
    """
    if angle == 0.0:
        return pil_img
    return F.affine(
        pil_img, angle=0.0, translate=[0, 0], scale=1.0, shear=[angle], fill=0
    )


# --- 4. Zoom ---
def apply_zoom(pil_img: Image.Image, zoom_factor: float) -> Image.Image:
    """
    Applies a deterministic zoom by a specific factor.

    Args:
        pil_img (Image.Image): The input PIL image.
        zoom_factor (float): The scaling factor. 1.0 is no zoom.
                             > 1.0 zooms in, < 1.0 zooms out.

    Returns:
        Image.Image: The zoomed image.
    """
    if zoom_factor == 1.0:
        return pil_img
    return F.affine(
        pil_img,
        angle=0.0,
        translate=[0, 0],
        scale=zoom_factor,
        shear=[0.0, 0.0],
        fill=0,
    )


# --- 5. Horizontal Flip ---
def apply_horizontal_flip(pil_img: Image.Image) -> Image.Image:
    """
    Applies a deterministic horizontal flip.

    Args:
        pil_img (Image.Image): The input PIL image.

    Returns:
        Image.Image: The horizontally flipped image.
    """
    return F.hflip(pil_img)


# --- 6. Vertical Flip ---
def apply_vertical_flip(pil_img: Image.Image) -> Image.Image:
    """
    Applies a deterministic vertical flip.

    Args:
        pil_img (Image.Image): The input PIL image.

    Returns:
        Image.Image: The vertically flipped image.
    """
    return F.vflip(pil_img)


# --- 7. Gaussian Noise ---
def add_gaussian_noise(
    pil_img: Image.Image, mean: float = 0.0, std: float = 25.0
) -> Image.Image:
    """
    Adds pixel-wise Gaussian noise to a PIL.Image.

    Args:
        pil_img (Image.Image): The input PIL image.
        mean (float): The mean of the Gaussian distribution.
        std (float): The standard deviation of the Gaussian distribution.

    Returns:
        Image.Image: A new PIL.Image object with added noise.
    """
    if std == 0.0 and mean == 0.0:
        return pil_img

    img_array = np.array(pil_img, dtype=np.float32)
    noise = np.random.normal(mean, std, img_array.shape)
    noisy_img_array = img_array + noise
    noisy_img_array = np.clip(noisy_img_array, 0, 255)
    noisy_pil_img = Image.fromarray(noisy_img_array.astype(np.uint8))
    return noisy_pil_img


# -------------------------------------------------
# MAIN
# -------------------------------------------------


def main():
    """
    Main function to load an image, apply a series of individual
    perturbations, save the results, and save a combined version.
    """
    # --- Configuration ---
    IMAGE_PATH = "/home1/adoyle2025/Distribution-Shift-Lane-Perception/datasets/CULane/driver_23_30frame/05151643_0420.MP4/00000.jpg"
    OUTPUT_DIR = "perturbation_examples"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving all perturbation examples to: {OUTPUT_DIR}/")

    # --- Define all shifts to apply ---
    shifts_to_apply = [
        GaussianShift(std=50.0, mean=0.0),
        RotationShift(angle=45.0),
        TranslationShift(width_shift_frac=0.2, height_shift_frac=0.2),
        ShearShift(angle=20.0),
        ZoomShift(zoom_factor=1.3),  # 30% zoom in
        HorizontalFlipShift(),
        VerticalFlipShift(),
        ZoomShift(zoom_factor=0.7),  # 30% zoom out
    ]

    # --- Main Logic ---
    try:
        with Image.open(IMAGE_PATH) as im:
            print(f"Loaded original image: {IMAGE_PATH} (Size: {im.size})")

            # Save the original for comparison
            im.save(os.path.join(OUTPUT_DIR, "0_original.png"))

            # --- 1. Apply and save each shift individually ---
            print("\n--- Applying Individual Shifts ---")
            for i, shift in enumerate(shifts_to_apply):

                # Apply the shift to the *original* image
                shifted_img = apply_shift(im, shift)

                # Create a filename
                filename = f"{i+1}_{shift.type.value.lower().replace(' ', '_')}.png"

                shifted_img.save(os.path.join(OUTPUT_DIR, filename))
                print(f"Saved {filename}")

            # --- 2. Apply and save a combined version ---
            print("\n--- Applying Combined Shift ---")
            combined_img = im
            for shift in shifts_to_apply:
                combined_img = apply_shift(combined_img, shift)

            combined_img.save(os.path.join(OUTPUT_DIR, "9_combined.png"))
            print("Saved 9_combined.png")

    except FileNotFoundError:
        print(f"ERROR: Could not find image file at {IMAGE_PATH}")
    except ImportError:
        print("\n--- PyTorch/Torchvision not found. ---")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
