import unittest
import os
import json
import shutil

# Import the class to be tested
from .data_logging import JsonExperimentManager, JsonDict


class TestJsonExperimentManager(unittest.TestCase):
    """
    Comprehensive unit test suite for the JsonExperimentManager class.

    This suite tests file creation, loading, data manipulation,
    and error handling.
    """

    def setUp(self) -> None:
        """
        Set up a clean test environment before each test.

        This creates a path for a temporary test directory and ensures
        it is clean before the test runs.
        """
        # Define a temporary directory for test files
        self.test_dir: str = "test_experiment_data_temp"
        self.test_filename: str = "test_results.json"
        self.test_filepath: str = os.path.join(self.test_dir, self.test_filename)

        # Clean up any lingering directories from previous failed runs
        self.tearDown()

    def tearDown(self) -> None:
        """
        Clean up the test environment after each test.

        This recursively removes the temporary test directory and all its
        contents.
        """
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _get_raw_file_content(self) -> JsonDict:
        """Helper function to read the test JSON file directly."""
        try:
            with open(self.test_filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            self.fail(f"Helper _get_raw_file_content failed: {e}")
            return {}

    def test_01_initialization_creates_new_file_and_dir(self) -> None:
        """
        Test if the directory and a valid empty JSON file are created
        when neither exists. (Requirement 2)
        """
        self.assertFalse(
            os.path.exists(self.test_dir), "Test dir should not exist at start"
        )

        manager = JsonExperimentManager(
            file_location=self.test_dir, file_name=self.test_filename
        )

        self.assertTrue(os.path.isdir(self.test_dir), "Directory was not created")
        self.assertTrue(os.path.isfile(self.test_filepath), "File was not created")

        # Check that the file content is a valid, empty structure
        content = self._get_raw_file_content()
        self.assertEqual(content, {"experiments": []})
        self.assertEqual(manager.get_all_experiments(), [])

    def test_02_initialization_loads_existing_file(self) -> None:
        """
        Test if the manager correctly loads data from a valid, existing
        JSON file.
        """
        # Manually create the directory and a valid file
        os.makedirs(self.test_dir)
        existing_data: JsonDict = {
            "experiments": [{"arguments": {"a": 1}, "data": {"b": 2}}]
        }
        with open(self.test_filepath, "w") as f:
            json.dump(existing_data, f)

        manager = JsonExperimentManager(
            file_location=self.test_dir, file_name=self.test_filename
        )

        # Check if the loaded data matches the pre-existing data
        loaded_experiments = manager.get_all_experiments()
        self.assertEqual(len(loaded_experiments), 1)
        self.assertEqual(loaded_experiments, existing_data["experiments"])

    def test_03_initialization_resets_empty_file(self) -> None:
        """
        Test if an existing but empty (0-byte) file is correctly
        handled and reset to the default structure.
        """
        os.makedirs(self.test_dir)
        # Create an empty file
        with open(self.test_filepath, "w") as f:
            pass

        self.assertEqual(
            os.path.getsize(self.test_filepath), 0, "File should be 0 bytes"
        )

        # This should print a warning and reset the file
        manager = JsonExperimentManager(
            file_location=self.test_dir, file_name=self.test_filename
        )

        self.assertGreater(
            os.path.getsize(self.test_filepath),
            0,
            "File should not be 0 bytes after reset",
        )
        content = self._get_raw_file_content()
        self.assertEqual(content, {"experiments": []})
        self.assertEqual(manager.get_all_experiments(), [])

    def test_04_initialization_resets_corrupt_json(self) -> None:
        """
        Test if a file with invalid JSON syntax is reset to the
        default structure.
        """
        os.makedirs(self.test_dir)
        # Write corrupt JSON
        with open(self.test_filepath, "w") as f:
            f.write("{'invalid': 'json', ")

        # This should print a warning and reset the file
        manager = JsonExperimentManager(
            file_location=self.test_dir, file_name=self.test_filename
        )

        content = self._get_raw_file_content()
        self.assertEqual(content, {"experiments": []})
        self.assertEqual(manager.get_all_experiments(), [])

    def test_05_initialization_resets_wrong_structure(self) -> None:
        """
        Test if a file with valid JSON but the wrong data structure
        is reset to the default structure.
        """
        os.makedirs(self.test_dir)
        wrong_data: JsonDict = {"some_other_key": "some_value"}
        with open(self.test_filepath, "w") as f:
            json.dump(wrong_data, f)

        # This should print a warning and reset the file
        manager = JsonExperimentManager(
            file_location=self.test_dir, file_name=self.test_filename
        )

        content = self._get_raw_file_content()
        self.assertEqual(content, {"experiments": []})
        self.assertEqual(manager.get_all_experiments(), [])

    def test_06_add_experiment_appends_data(self) -> None:
        """
        Test if add_experiment correctly appends data to the internal
        state and to the file. (Requirement 2)
        """
        manager = JsonExperimentManager(
            file_location=self.test_dir, file_name=self.test_filename
        )

        self.assertEqual(manager.get_all_experiments(), [], "Should start empty")

        # Add first experiment
        args1: JsonDict = {"voltage": 5, "material": "copper"}
        data1: JsonDict = {"resistance": 1.2}
        manager.add_experiment(args1, data1)

        # Check internal state
        self.assertEqual(len(manager.get_all_experiments()), 1)
        self.assertEqual(manager.get_all_experiments()[0]["arguments"], args1)

        # Check file state
        content1 = self._get_raw_file_content()
        self.assertEqual(len(content1["experiments"]), 1)
        self.assertEqual(content1["experiments"][0]["arguments"], args1)

        # Add second experiment
        args2: JsonDict = {"voltage": 10, "material": "silver"}
        data2: JsonDict = {"resistance": 0.8}
        manager.add_experiment(args2, data2)

        # Check internal state
        self.assertEqual(len(manager.get_all_experiments()), 2)
        self.assertEqual(manager.get_all_experiments()[1]["arguments"], args2)

        # Check file state
        content2 = self._get_raw_file_content()
        self.assertEqual(len(content2["experiments"]), 2)
        self.assertEqual(content2["experiments"][1]["arguments"], args2)

    def test_07_property_getters(self) -> None:
        """
        Test if the getter properties return the correct initialization
        values. (Requirement 1 & 3)
        """
        style = 2
        manager = JsonExperimentManager(
            file_location=self.test_dir, file_name=self.test_filename, style=style
        )

        self.assertEqual(manager.file_location, self.test_dir)
        self.assertEqual(manager.file_name, self.test_filename)
        self.assertEqual(manager.style, style)

    def test_08_property_setters_forbidden(self) -> None:
        """
        Test if changing file_location or file_name raises a ValueError.
        (Requirement 3)
        """
        manager = JsonExperimentManager(
            file_location=self.test_dir, file_name=self.test_filename
        )

        with self.assertRaisesRegex(ValueError, "Cannot change 'file_location'"):
            manager.file_location = "new_dir"

        with self.assertRaisesRegex(ValueError, "Cannot change 'file_name'"):
            manager.file_name = "new_file.json"

    def test_09_style_setter_rewrites_file(self) -> None:
        """
        Test if changing the style property rewrites the file with the
        new formatting. (Requirement 3)
        """
        # 1. Initialize with pretty-print style (indent=4)
        manager = JsonExperimentManager(
            file_location=self.test_dir, file_name=self.test_filename, style=4
        )
        manager.add_experiment({"a": 1}, {"b": 2})

        # Read raw content, it should have newlines and spaces
        with open(self.test_filepath, "r") as f:
            pretty_content = f.read()

        self.assertIn("\n", pretty_content)
        self.assertIn("    ", pretty_content)

        # 2. Change style to compact (indent=None)
        manager.style = None
        self.assertEqual(manager.style, None)

        # Read raw content, it should be compact
        with open(self.test_filepath, "r") as f:
            compact_content = f.read()

        self.assertLess(len(compact_content), len(pretty_content))
        # It should not have the 4-space indent
        self.assertNotIn("    ", compact_content)

        # 3. Change style back to pretty-print
        manager.style = 4
        self.assertEqual(manager.style, 4)

        # Read raw content, it should match the original pretty content
        with open(self.test_filepath, "r") as f:
            pretty_content_again = f.read()

        self.assertEqual(pretty_content_again, pretty_content)

    def test_10_style_setter_invalid_type(self) -> None:
        """
        Test if setting style to an invalid type raises a ValueError.
        """
        manager = JsonExperimentManager(
            file_location=self.test_dir, file_name=self.test_filename
        )

        with self.assertRaisesRegex(ValueError, "Style must be an integer or None"):
            manager.style = "four"  # type: ignore

        with self.assertRaisesRegex(ValueError, "Style must be an integer or None"):
            manager.style = 3.14  # type: ignore

    def test_11_style_setter_no_rewrite_if_same_value(self) -> None:
        """
        Test that the file is not rewritten if the style is set to
        its existing value. (This is an optimization test based on the
        implementation `if self._style != value:`)

        We test this by monitoring the file's modification time.
        """
        manager = JsonExperimentManager(
            file_location=self.test_dir, file_name=self.test_filename, style=4
        )
        manager.add_experiment({"a": 1}, {"b": 2})

        # Get the last modification time
        last_mod_time = os.path.getmtime(self.test_filepath)

        # Set style to the same value
        manager.style = 4

        # Get new modification time
        new_mod_time = os.path.getmtime(self.test_filepath)

        # They should be identical, as no write should have occurred
        self.assertEqual(
            last_mod_time, new_mod_time, "File was rewritten unnecessarily"
        )


if __name__ == "__main__":
    unittest.main()
