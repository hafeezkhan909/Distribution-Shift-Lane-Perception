# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import numpy as np
from enum import Enum
from math import ceil
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
import os
from abc import ABC, abstractmethod


# -------------------------------------------------
# SHIFT TYPES
# -------------------------------------------------
class ShiftTypes(Enum):
    GAUSSIAN = "Gaussian"
    IMAGE_GENERATOR = "Image Generator"


# -------------------------------------------------
# SHIFT APPLICATOR
# -------------------------------------------------


def apply_shift(image, shift, mean, std):
    if shift == ShiftTypes.GAUSSIAN:
        # amount: 100, 10, 1
        print(f"GN shift: (Mean: {mean}, Std:{std})")
        return add_gaussian_noise(image, mean, std)
    # elif shift == ShiftTypes.IMAGE_GENERATOR:
    #     # amount: 10, 40, 90
    #     print(f'GN shift: (Mean: {mean}, Std:{std})')
    #     return image_generator(image, mean, std)


# -------------------------------------------------
# DATA UTILS
# -------------------------------------------------


# --- 1. Rotation ---
def apply_rotation(pil_img: Image.Image, rot_range: float) -> Image.Image:
    """Applies a random rotation between [-rot_range, +rot_range] degrees."""
    if rot_range == 0.0:
        return pil_img
    # fill=0 sets the background to black
    transformer = transforms.RandomRotation(degrees=(-rot_range, rot_range), fill=0)
    return transformer(pil_img)


# --- 2. Translation (Width/Height Shift) ---
def apply_translation(
    pil_img: Image.Image, width_range: float, height_range: float
) -> Image.Image:
    """Applies a random horizontal and vertical shift."""
    if width_range == 0.0 and height_range == 0.0:
        return pil_img
    # translate wants a (width_fraction, height_fraction) tuple
    transformer = transforms.RandomAffine(
        degrees=0, translate=(width_range, height_range), fill=0
    )
    return transformer(pil_img)


# --- 3. Shear ---
def apply_shear(pil_img: Image.Image, shear_range: float) -> Image.Image:
    """Applies a random shear between [-shear_range, +shear_range] degrees."""
    if shear_range == 0.0:
        return pil_img
    transformer = transforms.RandomAffine(
        degrees=0, shear=(-shear_range, shear_range), fill=0
    )
    return transformer(pil_img)


# --- 4. Zoom ---
def apply_zoom(pil_img: Image.Image, zoom_range: float) -> Image.Image:
    """Applies a random zoom. zoom_range=0.2 means 80% to 120% zoom."""
    if zoom_range == 0.0:
        return pil_img
    scale_param = (1.0 - zoom_range, 1.0 + zoom_range)
    transformer = transforms.RandomAffine(degrees=0, scale=scale_param, fill=0)
    return transformer(pil_img)


# --- 5. Horizontal Flip ---
def apply_horizontal_flip(pil_img: Image.Image) -> Image.Image:
    """Applies a horizontal flip."""
    return F.hflip(pil_img)


# --- 6. Vertical Flip ---
def apply_vertical_flip(pil_img: Image.Image) -> Image.Image:
    """Applies a vertical flip."""
    return F.vflip(pil_img)


def add_gaussian_noise(
    pil_img: Image.Image, mean: float = 0.0, std: float = 25.0
) -> Image.Image:
    """
    Adds Gaussian noise to a PIL.Image.

    :param pil_img: The PIL.Image object.
    :param mean: The mean of the Gaussian distribution.
    :param std: The standard deviation of the Gaussian distribution.
    :return: A new PIL.Image object with added noise.
    """

    # 1. Convert PIL image to NumPy array
    img_array = np.array(pil_img, dtype=np.float32)

    # 2. Generate Gaussian noise
    noise = np.random.normal(mean, std, img_array.shape)

    # 3. Add noise to the image array
    noisy_img_array = img_array + noise

    # 4. Clip values to stay in the valid [0, 255] range
    noisy_img_array = np.clip(noisy_img_array, 0, 255)

    # 5. Convert back to a PIL Image
    noisy_pil_img = Image.fromarray(noisy_img_array.astype(np.uint8))
    return noisy_pil_img


def perturb_image(
    pil_img: Image.Image,
    rot_range: float = 0.0,
    width_range: float = 0.0,
    height_range: float = 0.0,
    shear_range: float = 0.0,
    zoom_range: float = 0.0,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
) -> Image.Image:
    """
    Applies a series of augmentations by calling each 'baby function'
    in a specific order.
    """

    # You can change the order of operations by re-ordering this list
    img_out = pil_img

    # 1. Apply Horizontal Flip
    if horizontal_flip:
        img_out = apply_horizontal_flip(pil_img)

    # 2. Apply Vertical Flip
    if vertical_flip:
        img_out = apply_vertical_flip(img_out)

    # 3. Apply Rotation
    img_out = apply_rotation(img_out, rot_range)

    # 4. Apply Translation
    img_out = apply_translation(img_out, width_range, height_range)

    # 5. Apply Shear
    img_out = apply_shear(img_out, shear_range)

    # 6. Apply Zoom
    img_out = apply_zoom(img_out, zoom_range)

    return img_out


def main():
    # --- Configuration ---
    IMAGE_PATH = "/home1/adoyle2025/Distribution-Shift-Lane-Perception/datasets/CULane/driver_23_30frame/05151643_0420.MP4/00000.jpg"
    OUTPUT_DIR = "perturbation_examples"

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving all perturbation examples to: {OUTPUT_DIR}/")

    # --- Parameters for Demo ---
    ROT_DEGREES = 45.0
    TRANS_RANGE = 0.2
    SHEAR_DEGREES = 20.0
    ZOOM_RANGE = 0.3
    NOISE_STD = 50.0

    # --- Main Logic ---
    try:
        with Image.open(IMAGE_PATH) as im:
            a = "test1"
            print(f"Loaded original image: {IMAGE_PATH} (Size: {im.size})")

            # Save the original for comparison
            im.save(os.path.join(OUTPUT_DIR, "0_original.png"))

            # 1. Apply Rotation
            rotated_img = apply_rotation(im, ROT_DEGREES)
            rotated_img.save(os.path.join(OUTPUT_DIR, "1_rotated.png"))
            print(f"{a}_rotated_{ROT_DEGREES}.png")

            # 2. Apply Translation
            translated_img = apply_translation(im, TRANS_RANGE, TRANS_RANGE)
            translated_img.save(os.path.join(OUTPUT_DIR, "2_translated.png"))
            print(f"{a}_translated_range{TRANS_RANGE}.png")

            # 3. Apply Shear
            sheared_img = apply_shear(im, SHEAR_DEGREES)
            sheared_img.save(os.path.join(OUTPUT_DIR, "3_sheared.png"))
            print(f"{a}_sheared_deg{SHEAR_DEGREES}.png")

            # 4. Apply Zoom
            zoomed_img = apply_zoom(im, ZOOM_RANGE)
            zoomed_img.save(os.path.join(OUTPUT_DIR, "4_zoomed.png"))
            print(f"{a}_zoomed_range{ZOOM_RANGE}.png")

            # 5. Apply Horizontal Flip
            h_flipped_img = apply_horizontal_flip(im)
            h_flipped_img.save(os.path.join(OUTPUT_DIR, "5_h_flipped.png"))
            print(f"{a}_h_flipped.png")

            # 6. Apply Vertical Flip
            v_flipped_img = apply_vertical_flip(im)
            v_flipped_img.save(os.path.join(OUTPUT_DIR, "6_v_flipped.png"))
            print(f"{a}_v_flipped.png")

            # 7. Apply Gaussian Noise (using apply_shift)
            noisy_img = apply_shift(im, ShiftTypes.GAUSSIAN, mean=0, std=NOISE_STD)
            noisy_img.save(os.path.join(OUTPUT_DIR, "7_noisy.png"))
            print(f"{a}_noisy_std{NOISE_STD}.png")

            # 8. Apply Combined Perturbation (from original main)
            combined_img = perturb_image(
                im,
                rot_range=ROT_DEGREES,
                width_range=TRANS_RANGE,
                height_range=TRANS_RANGE,
                shear_range=SHEAR_DEGREES,
                zoom_range=ZOOM_RANGE,
                horizontal_flip=True,
                vertical_flip=True,
            )
            combined_img.save(os.path.join(OUTPUT_DIR, "8_combined.png"))
            print(f"{a}_combined.png")  # <-- Fixed f-string

    except FileNotFoundError:
        print(f"ERROR: Could not find image file at {IMAGE_PATH}")
    except ImportError:
        print("\n--- PyTorch/Torchvision not found. ---")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
