import unittest
import numpy as np
from PIL import Image

# Import all the functions and classes to be tested
from data.data_utils import (
    DataShift,
    GaussianShift,
    RotationShift,
    TranslationShift,
    ShearShift,
    ZoomShift,
    HorizontalFlipShift,
    VerticalFlipShift,
    apply_shift,
    add_gaussian_noise,
    apply_rotation,
    apply_translation,
    apply_shear,
    apply_zoom,
    apply_horizontal_flip,
    apply_vertical_flip,
)


class TestDataUtils(unittest.TestCase):

    def setUp(self):
        """
        Set up a consistent test image before each test method.
        This creates a 100x100 RGB image with a solid color
        AND a white cross, so transforms are detectable.
        """
        self.size = (100, 100)
        self.color = (50, 100, 150)
        self.image = Image.new("RGB", self.size, self.color)

        # Add a white cross to make transforms visible
        white = (255, 255, 255)
        for i in range(self.size[0]):
            self.image.putpixel((i, self.size[1] // 2), white)  # Horizontal line
            self.image.putpixel((self.size[0] // 2, i), white)  # Vertical line

        self.image_array = np.array(self.image)

    def assertImagesNotEqual(self, img1_arr, img2_arr):
        """Helper assertion to check that two image arrays are NOT identical."""
        self.assertFalse(
            np.array_equal(img1_arr, img2_arr), "Images were unexpectedly identical."
        )

    def assertImagesEqual(self, img1_arr, img2_arr):
        """Helper assertion to check that two image arrays ARE identical."""
        self.assertTrue(
            np.array_equal(img1_arr, img2_arr), "Images were unexpectedly different."
        )

    def assertSameShape(self, img1, img2):
        """Helper assertion to check two PIL images have the same size."""
        self.assertEqual(img1.size, img2.size, "Image sizes do not match.")

    # --- Test Individual "Baby" Functions ---

    def test_add_gaussian_noise_changes_image(self):
        """Test that non-zero std noise changes the image."""
        noisy_img = add_gaussian_noise(self.image, mean=0, std=50)
        noisy_arr = np.array(noisy_img)
        self.assertSameShape(self.image, noisy_img)
        self.assertImagesNotEqual(self.image_array, noisy_arr)

    def test_add_gaussian_noise_zero_std(self):
        """Test that zero-std noise does NOT change the image."""
        noisy_img = add_gaussian_noise(self.image, mean=0, std=0)
        noisy_arr = np.array(noisy_img)
        self.assertSameShape(self.image, noisy_img)
        self.assertImagesEqual(self.image_array, noisy_arr)

    def test_apply_rotation_zero(self):
        """Test that angle=0 does not change the image."""
        rotated_img = apply_rotation(self.image, 0.0)
        self.assertImagesEqual(self.image_array, np.array(rotated_img))

    def test_apply_rotation_nonzero(self):
        """Test that angle>0 DOES change the image."""
        rotated_img = apply_rotation(self.image, 45.0)
        self.assertSameShape(self.image, rotated_img)
        self.assertImagesNotEqual(self.image_array, np.array(rotated_img))

    def test_apply_translation_zero(self):
        """Test that frac=0 does not change the image."""
        translated_img = apply_translation(self.image, 0.0, 0.0)
        self.assertImagesEqual(self.image_array, np.array(translated_img))

    def test_apply_translation_nonzero(self):
        """Test that frac>0 DOES change the image."""
        translated_img = apply_translation(self.image, 0.2, 0.2)
        self.assertSameShape(self.image, translated_img)
        self.assertImagesNotEqual(self.image_array, np.array(translated_img))

    def test_apply_shear_zero(self):
        """Test that angle=0 does not change the image."""
        sheared_img = apply_shear(self.image, 0.0)
        self.assertImagesEqual(self.image_array, np.array(sheared_img))

    def test_apply_shear_nonzero(self):
        """Test that angle>0 DOES change the image."""
        sheared_img = apply_shear(self.image, 20.0)
        self.assertSameShape(self.image, sheared_img)
        self.assertImagesNotEqual(self.image_array, np.array(sheared_img))

    def test_apply_zoom_one(self):
        """Test that zoom_factor=1.0 (default) does not change the image."""
        zoomed_img = apply_zoom(self.image, 1.0)
        self.assertImagesEqual(self.image_array, np.array(zoomed_img))

    def test_apply_zoom_factor(self):
        """Test that zoom_factor != 1.0 DOES change the image."""
        zoomed_img = apply_zoom(self.image, 0.8)  # Zoom in
        self.assertSameShape(self.image, zoomed_img)
        self.assertImagesNotEqual(self.image_array, np.array(zoomed_img))

    def test_apply_horizontal_flip(self):
        """Test that the image flips horizontally."""
        flipped_img = apply_horizontal_flip(self.image)
        self.assertSameShape(self.image, flipped_img)
        self.assertImagesNotEqual(self.image_array, np.array(flipped_img))

    def test_apply_vertical_flip(self):
        """Test that vertical flips the image."""
        flipped_img = apply_vertical_flip(self.image)
        self.assertSameShape(self.image, flipped_img)
        self.assertImagesNotEqual(self.image_array, np.array(flipped_img))

    # --- Test Main 'apply_shift' Dispatcher Function ---

    def test_apply_shift_gaussian(self):
        """Test the apply_shift dispatcher with GaussianShift."""
        shift_obj = GaussianShift(std=50.0)
        shifted_img = apply_shift(self.image, shift_obj)
        self.assertImagesNotEqual(self.image_array, np.array(shifted_img))

    def test_apply_shift_rotation(self):
        """Test the apply_shift dispatcher with RotationShift."""
        shift_obj = RotationShift(angle=30.0)
        shifted_img = apply_shift(self.image, shift_obj)
        self.assertImagesNotEqual(self.image_array, np.array(shifted_img))

    def test_apply_shift_translation(self):
        """Test the apply_shift dispatcher with TranslationShift."""
        shift_obj = TranslationShift(width_shift_frac=0.1)
        shifted_img = apply_shift(self.image, shift_obj)
        self.assertImagesNotEqual(self.image_array, np.array(shifted_img))

    def test_apply_shift_shear(self):
        """Test the apply_shift dispatcher with ShearShift."""
        shift_obj = ShearShift(angle=15.0)
        shifted_img = apply_shift(self.image, shift_obj)
        self.assertImagesNotEqual(self.image_array, np.array(shifted_img))

    def test_apply_shift_zoom(self):
        """Test the apply_shift dispatcher with ZoomShift."""
        shift_obj = ZoomShift(zoom_factor=0.8)  # Zoom out
        shifted_img = apply_shift(self.image, shift_obj)
        self.assertImagesNotEqual(self.image_array, np.array(shifted_img))

    def test_apply_shift_hflip(self):
        """Test the apply_shift dispatcher with HorizontalFlipShift."""
        shift_obj = HorizontalFlipShift()
        shifted_img = apply_shift(self.image, shift_obj)
        self.assertImagesNotEqual(self.image_array, np.array(shifted_img))

    def test_apply_shift_vflip(self):
        """Test the apply_shift dispatcher with VerticalFlipShift."""
        shift_obj = VerticalFlipShift()
        shifted_img = apply_shift(self.image, shift_obj)
        self.assertImagesNotEqual(self.image_array, np.array(shifted_img))

    def test_apply_shift_unknown_type(self):
        """Test that an unknown shift type returns the original image."""

        # Create a dummy unknown shift class for testing
        class UnknownShift(DataShift):
            def __init__(self):
                self.type = "UNKNOWN_TYPE"

            def __str__(self):
                return "UnknownShift"

        shift_obj = UnknownShift()
        shifted_img = apply_shift(self.image, shift_obj)
        # The function should print a warning and return the original image
        self.assertImagesEqual(self.image_array, np.array(shifted_img))


if __name__ == "__main__":
    unittest.main()
