# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import numpy as np
from enum import Enum
from math import ceil
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from PIL import Image

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
        print(f'GN shift: (Mean: {mean}, Std:{std})')
        return add_gaussian_noise(image, mean, std)
    # elif shift == ShiftTypes.IMAGE_GENERATOR:
    #     # amount: 10, 40, 90
    #     print(f'GN shift: (Mean: {mean}, Std:{std})')
    #     return image_generator(image, mean, std)

# -------------------------------------------------
# DATA UTILS
# -------------------------------------------------

# Perform image perturbations.
# def image_generator(x, orig_dims, rot_range, width_range, height_range, shear_range, zoom_range, horizontal_flip,
#                     vertical_flip, delta=1.0):
#     indices = np.random.choice(x.shape[0], ceil(x.shape[0] * delta), replace=False)
#     datagen = ImageDataGenerator(rotation_range=rot_range,
#                                  width_shift_range=width_range,
#                                  height_shift_range=height_range,
#                                  shear_range=shear_range,
#                                  zoom_range=zoom_range,
#                                  horizontal_flip=horizontal_flip,
#                                  vertical_flip=vertical_flip,
#                                  fill_mode="nearest")
#     x_mod = x[indices, :]
#     for idx in range(len(x_mod)):
#         img_sample = x_mod[idx, :].reshape(orig_dims)
#         mod_img_sample = datagen.flow(np.array([img_sample]), batch_size=1)[0]
#         x_mod[idx, :] = mod_img_sample.reshape(np.prod(mod_img_sample.shape))
#     x[indices, :] = x_mod

#     return x, indices

def add_gaussian_noise(pil_img, mean=0, std=25):
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

def main():
    with Image.open("/home1/adoyle2025/Distribution-Shift-Lane-Perception/datasets/CULane/driver_23_30frame/05151643_0420.MP4/00000.jpg") as im:
        # Apply Shift of 1
        im_shifted1 = apply_shift(im, shift=ShiftTypes.GAUSSIAN, mean=0, std=1)
        im_shifted1.save("/home1/adoyle2025/Distribution-Shift-Lane-Perception/test1.jpeg", "JPEG")
        
        # Apply Shift of 10
        im_shifted10 = apply_shift(im, shift=ShiftTypes.GAUSSIAN, mean=0, std=10)
        im_shifted10.save("/home1/adoyle2025/Distribution-Shift-Lane-Perception/test10.jpeg", "JPEG")

        # Apply Shift of 100
        im_shifted100 = apply_shift(im, shift=ShiftTypes.GAUSSIAN, mean=0, std=100)
        im_shifted100.save("/home1/adoyle2025/Distribution-Shift-Lane-Perception/test100.jpeg", "JPEG")

if __name__ == "__main__":
    main()
