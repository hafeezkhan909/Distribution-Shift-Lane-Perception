# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
import os

# -------------------------------------------------
# SHIFT APPLICATOR
# -------------------------------------------------


def apply_shift(
    image: Image.Image,
    gaussian_sigma: float = 0.0,
    rotation_angle: float = 0,
    width_shift_frac: float = 0,
    height_shift_frac: float = 0,
    shear_angle: float = 0,
    zoom_factor: float = 1.0,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
) -> Image.Image:
    """
    Dispatcher function that applies a specific DataShift to a PIL image.

    Args:
        image (Image.Image): The input PIL image.
        dataShift (DataShift): An object (e.g., GaussianShift, RotationShift)
                                defining the shift to apply.

    Returns:
        Image.Image: The transformed PIL image.
    """

    if gaussian_sigma != 0.0:
        image = add_gaussian_noise(image, mean=0, std=gaussian_sigma)
    if rotation_angle != 0:
        image = apply_rotation(image, angle=rotation_angle)
    if width_shift_frac != 0 and height_shift_frac != 0:
        image = apply_translation(
            image,
            width_shift_frac=width_shift_frac,
            height_shift_frac=height_shift_frac,
        )
    elif width_shift_frac != 0:
        image = apply_translation(
            image,
            width_shift_frac=width_shift_frac,
            height_shift_frac=0,
        )
    elif height_shift_frac != 0:
        image = apply_translation(
            image,
            width_shift_frac=0,
            height_shift_frac=height_shift_frac,
        )
    if shear_angle != 0:
        image = apply_shear(image, angle=shear_angle)
    if zoom_factor != 1.0:
        image = apply_zoom(image, zoom_factor=zoom_factor)
    if horizontal_flip:
        image = apply_horizontal_flip(image)
    if vertical_flip:
        image = apply_vertical_flip(image)
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
    """Main function to load an image, apply a series of individual

    perturbations, save the results, and save a combined version.
    """
    # --- Configuration ---
    IMAGE_PATH = "/home1/adoyle2025/Datasets/Datasets/CULane/driver_100_30frame/05251312_0398.MP4/01770.jpg"
    OUTPUT_DIR = "perturbation_examples"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving all perturbation examples to: {OUTPUT_DIR}/")

    # --- Define all shifts to apply as dictionaries (kwargs for apply_shift) ---
    shifts_to_apply = [
        {"name": "gaussian_blur", "gaussian_sigma": 50.0},
        {"name": "rotation", "rotation_angle": 45.0},
        {
            "name": "translation",
            "width_shift_frac": 0.2,
            "height_shift_frac": 0.2,
        },
        {"name": "shear", "shear_angle": 20.0},
        {"name": "zoom_in", "zoom_factor": 1.3},
        {"name": "horizontal_flip", "horizontal_flip": True},
        {"name": "vertical_flip", "vertical_flip": True},
        {"name": "zoom_out", "zoom_factor": 0.7},
    ]

    # --- Main Logic ---
    try:
        with Image.open(IMAGE_PATH) as im:
            print(f"Loaded original image: {IMAGE_PATH} (Size: {im.size})")

            # Save the original for comparison
            im.save(os.path.join(OUTPUT_DIR, "0_original.png"))

            # --- 1. Apply and save each shift individually ---
            print("\n--- Applying Individual Shifts ---")
            for i, shift_config in enumerate(shifts_to_apply):

                # Copy to avoid modifying our original list
                kwargs = shift_config.copy()
                display_name = kwargs.pop("name")

                # **kwargs unpacks the dictionary directly into apply_shift parameters
                shifted_img = apply_shift(im, **kwargs)

                # Create a clean filename
                filename = f"{i+1}_{display_name}.png"
                shifted_img.save(os.path.join(OUTPUT_DIR, filename))
                print(f"Saved {filename}")

            # --- 2. Apply and save a combined version ---
            print("\n--- Applying Combined Shift ---")
            combined_img = im
            for shift_config in shifts_to_apply:
                kwargs = shift_config.copy()
                kwargs.pop("name")
                combined_img = apply_shift(combined_img, **kwargs)

            combined_img.save(os.path.join(OUTPUT_DIR, "9_combined.png"))
            print("Saved 9_combined.png")

    except FileNotFoundError:
        print(f"ERROR: Could not find image file at {IMAGE_PATH}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
