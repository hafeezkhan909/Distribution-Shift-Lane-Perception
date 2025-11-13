# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import numpy as np
from math import ceil
from keras.src.legacy.preprocessing.image import ImageDataGenerator


# -------------------------------------------------
# SHIFT APPLICATOR
# -------------------------------------------------


def apply_shift(X_te_orig, y_te_orig, shift, orig_dims, dataset):
    X_te_1 = None
    y_te_1 = None

    if shift == 'large_gn_shift_1.0':
        print('Large GN shift')
        normalization = 255.0
        X_te_1, _ = gaussian_noise_subset(X_te_orig, 100.0, normalization=normalization, delta_total=1.0)
        y_te_1 = y_te_orig.copy()
    elif shift == 'medium_gn_shift_1.0':
        print('Medium GN Shift')
        normalization = 255.0
        X_te_1, _ = gaussian_noise_subset(X_te_orig, 10.0, normalization=normalization, delta_total=1.0)
        y_te_1 = y_te_orig.copy()
    elif shift == 'small_gn_shift_1.0':
        print('Small GN Shift')
        normalization = 255.0
        X_te_1, _ = gaussian_noise_subset(X_te_orig, 1.0, normalization=normalization, delta_total=1.0)
        y_te_1 = y_te_orig.copy()
    elif shift == 'small_img_shift_1.0':
        print('Small image shift')
        X_te_1, _ = image_generator(X_te_orig, orig_dims, 10, 0.05, 0.05, 0.1, 0.1, False, False, delta=1.0)
        y_te_1 = y_te_orig.copy()
    elif shift == 'medium_img_shift_1.0':
        print('Medium image shift')
        X_te_1, _ = image_generator(X_te_orig, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=1.0)
        y_te_1 = y_te_orig.copy()
    elif shift == 'large_img_shift_1.0':
        print('Large image shift')
        X_te_1, _ = image_generator(X_te_orig, orig_dims, 90, 0.4, 0.4, 0.3, 0.4, True, True, delta=1.0)
        y_te_1 = y_te_orig.copy()

    return (X_te_1, y_te_1)

# -------------------------------------------------
# DATA UTILS
# -------------------------------------------------

# Perform image perturbations.
def image_generator(x, orig_dims, rot_range, width_range, height_range, shear_range, zoom_range, horizontal_flip,
                    vertical_flip, delta=1.0):
    indices = np.random.choice(x.shape[0], ceil(x.shape[0] * delta), replace=False)
    datagen = ImageDataGenerator(rotation_range=rot_range,
                                 width_shift_range=width_range,
                                 height_shift_range=height_range,
                                 shear_range=shear_range,
                                 zoom_range=zoom_range,
                                 horizontal_flip=horizontal_flip,
                                 vertical_flip=vertical_flip,
                                 fill_mode="nearest")
    x_mod = x[indices, :]
    for idx in range(len(x_mod)):
        img_sample = x_mod[idx, :].reshape(orig_dims)
        mod_img_sample = datagen.flow(np.array([img_sample]), batch_size=1)[0]
        x_mod[idx, :] = mod_img_sample.reshape(np.prod(mod_img_sample.shape))
    x[indices, :] = x_mod

    return x, indices


def gaussian_noise_subset(x, noise_amt, normalization=1.0, delta_total=1.0, clip=True):
    indices = np.random.choice(x.shape[0], ceil(x.shape[0] * delta_total), replace=False)
    x_mod = x[indices, :]
    noise = np.random.normal(0, noise_amt / normalization, (x_mod.shape[0], x_mod.shape[1]))
    if clip:
        x_mod = np.clip(x_mod + noise, 0., 1.)
    else:
        x_mod = x_mod + noise
    x[indices, :] = x_mod
    return x, indices
