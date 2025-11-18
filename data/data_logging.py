import json
import os
from typing import Any, Dict, List, Union

# Define type aliases for clarity in type hinting
JsonDict = Dict[str, Any]
ExperimentList = List[JsonDict]
JsonStyle = Union[int, None]


class JsonExperimentManager:
    """Manages a JSON file for storing and retrieving experimental data.

    This class provides a high-level interface for handling a single JSON file
    that acts as a database for a series of experiments. It ensures that the
    file is correctly structured and handles file creation, reading, and
    atomic updates when adding new data.

    The expected JSON structure is:
    {
      "experiments": [
        {
          "arguments": { "key": "value", ... },
          "data": { "key": "value", ... }
        },
        ...
      ]
    }

    Attributes:
        file_location (str): Get-only. The directory path for the JSON file.
        file_name (str): Get-only. The file name for the JSON file.
        style (JsonStyle): Get/Set. The indentation style for the JSON file.
    """

    def __init__(self, file_location: str, file_name: str, style: JsonStyle = 4):
        """Initializes the experiment manager.

        This will either load an existing experiment file or create a new one
        if it does not exist at the specified path. It also ensures the
        containing directory exists.

        Args:
            file_location (str): The directory path where the JSON file is
                or should be stored.
            file_name (str): The name of the JSON file (e.g., "results.json").
            style (JsonStyle): The indentation style for the JSON file, passed
                to `json.dump()`. Defaults to 4 (pretty-print).
                Use `None` for a compact, single-line file.

        Raises:
            OSError: If the directory cannot be created at `file_location`.
        """
        self._file_location: str = file_location
        self._file_name: str = file_name
        self._style: JsonStyle = style

        # Ensure the directory exists
        if not os.path.exists(self._file_location):
            try:
                os.makedirs(self._file_location)
                print(f"Created directory: {self._file_location}")
            except OSError as e:
                print(f"Error creating directory {self._file_location}: {e}")
                raise  # Re-raise the exception as it's a critical failure

        self._file_path: str = os.path.join(self._file_location, self._file_name)

        # Internal data store, initialized with the default structure
        self._data: JsonDict = {"experiments": []}

        self._load_or_create_file()

    def _load_or_create_file(self) -> None:
        """Loads data from file or creates a new file.

        Attempts to read and parse the JSON file. If the file is not found,
        it creates a new one. If the file is found but is empty, corrupt,
        or improperly structured, it will be overwritten with the
        default empty structure.
        """
        try:
            with open(self._file_path, "r") as f:
                loaded_data = json.load(f)

                # Basic structure validation
                if (
                    isinstance(loaded_data, dict)
                    and "experiments" in loaded_data
                    and isinstance(loaded_data["experiments"], list)
                ):

                    self._data = loaded_data
                else:
                    print(
                        f"Warning: File {self._file_path} has invalid structure. Resetting."
                    )
                    self._data = {"experiments": []}
                    self._save_to_file()  # Overwrite invalid file

        except FileNotFoundError:
            # Requirement 2: File doesn't exist, so create it
            print(f"File not found. Creating new file at {self._file_path}")
            self._save_to_file()
        except json.JSONDecodeError:
            # File is empty or corrupt
            print(f"Warning: File {self._file_path} is empty or corrupt. Resetting.")
            self._data = {"experiments": []}
            self._save_to_file()

    def _save_to_file(self) -> None:
        """Saves the current internal data state to the JSON file.

        This method overwrites the entire file with the data held in
        `self._data`, using the indentation style specified in `self._style`.

        Raises:
            IOError: If the file cannot be written to (e.g., due to
                permissions issues).
            TypeError: If the data in `self._data` is not JSON-serializable.
        """
        try:
            with open(self._file_path, "w") as f:
                json.dump(self._data, f, indent=self._style)
        except (IOError, TypeError) as e:
            print(f"Error: Could not write to file {self._file_path}. {e}")
            raise

    # --- Public Methods ---

    def add_experiment(self, arguments: JsonDict, data: JsonDict) -> None:
        """Adds a new experiment record and saves it to the file.

        This method appends a new experiment (composed of 'arguments' and 'data')
        to the internal list and then immediately persists the entire updated
        dataset to the JSON file.

        Args:
            arguments (JsonDict): A dictionary of independent variables
                (e.g., {"voltage": 5, "material": "copper"}).
            data (JsonDict): A dictionary of dependent variables or results
                (e.g., {"resistance": 1.2, "time_ms": 150}).
        """
        new_experiment: JsonDict = {"arguments": arguments, "data": data}

        # Ensure 'experiments' key exists, just in case
        if "experiments" not in self._data or not isinstance(
            self._data["experiments"], list
        ):
            self._data["experiments"] = []

        # Requirement 2: If file exists, just add to the data
        self._data["experiments"].append(new_experiment)
        self._save_to_file()
        print(
            f"Added new experiment. Total experiments: {len(self._data['experiments'])}"
        )

    def get_all_experiments(self) -> ExperimentList:
        """Retrieves all experiments from the data.

        Returns:
            ExperimentList: A list of all experiment objects. Each object
                is a dictionary, typically with 'arguments' and 'data' keys.
                Returns an empty list if no experiments are present.
        """
        return self._data.get("experiments", [])

    # --- Getters and Setters (Requirement 3) ---

    @property
    def file_location(self) -> str:
        """Get the file location (directory).

        Returns:
            str: The absolute or relative path to the directory containing
                 the JSON file.
        """
        return self._file_location

    @file_location.setter
    def file_location(self, value: str) -> None:
        """Setting file location after initialization is forbidden.

        Raises:
            ValueError: Always, as this property is read-only after init.
        """
        raise ValueError(
            "Cannot change 'file_location' after initialization. "
            "Create a new manager instance instead."
        )

    @property
    def file_name(self) -> str:
        """Get the file name.

        Returns:
            str: The name of the JSON file.
        """
        return self._file_name

    @file_name.setter
    def file_name(self, value: str) -> None:
        """Setting file name after initialization is forbidden.

        Raises:
            ValueError: Always, as this property is read-only after init.
        """
        raise ValueError(
            "Cannot change 'file_name' after initialization. "
            "Create a new manager instance instead."
        )

    @property
    def style(self) -> JsonStyle:
        """Get the JSON indentation style.

        Returns:
            JsonStyle (int | None): The current indentation level used for
                                    JSON formatting.
        """
        return self._style

    @style.setter
    def style(self, value: JsonStyle) -> None:
        """Sets the JSON indentation style.

        This will immediately re-save the entire JSON file with the new
        formatting if the new style is different from the old one.

        Args:
            value (JsonStyle): The new indentation style (e.g., 4, 2, or None).

        Raises:
            ValueError: If the provided style is not an integer or None.
            IOError: If the file rewrite fails (propagated from `_save_to_file`).
            TypeError: If the rewrite fails (propagated from `_save_to_file`).
        """
        if not (isinstance(value, int) or value is None):
            raise ValueError("Style must be an integer or None.")

        if self._style != value:
            old_style = self._style
            self._style = value
            print(
                f"Style changed from {old_style} to {value}. Rewriting file {self._file_path}..."
            )
            try:
                self._save_to_file()
                print("File rewrite complete.")
            except (IOError, TypeError) as e:
                # Revert style if save fails to maintain consistent state
                self._style = old_style
                print(f"Failed to rewrite file with new style: {e}. Style reverted.")
                raise


if __name__ == "__main__":
    # --- Example Usage ---
    # This block will only run when the script is executed directly.

    # Define a test directory in the current working directory
    test_dir = os.path.join(os.getcwd(), "experiment_data_prod")

    # --- 1. Initialize the manager ---
    print("--- Initializing Manager ---")
    try:
        manager = JsonExperimentManager(
            file_location=test_dir,
            file_name="prod_results.json",
            style=4,  # Use 4-space indentation
        )

        # --- 2. Add some experiments ---
        print("\n--- Adding Experiments ---")
        manager.add_experiment(
            arguments={"voltage": 5, "material": "copper"},
            data={"resistance": 1.2, "time_ms": 150},
        )
        manager.add_experiment(
            arguments={"voltage": 10, "material": "copper"},
            data={"resistance": 1.1, "time_ms": 145},
        )

        # --- 3. Get all data ---
        print("\n--- Retrieving Data ---")
        all_data: ExperimentList = manager.get_all_experiments()
        print(f"Retrieved {len(all_data)} experiments.")
        # Print it nicely
        print(json.dumps(all_data, indent=2))

        # --- 4. Change the style ---
        print("\n--- Changing Style ---")
        # Set style to None for a compact file
        manager.style = None
        print(f"Current style: {manager.style}")

        # Check 'experiment_data_prod/prod_results.json' to see it's now compact.
        # Let's change it back to pretty-print
        manager.style = 4
        print(f"Current style: {manager.style}")
        # The file will be rewritten again.

        # --- 5. Test the error-throwing setters ---
        print("\n--- Testing Forbidden Setters ---")
        try:
            manager.file_name = "new_results.json"
        except ValueError as e:
            print(f"Caught expected error: {e}")

        try:
            manager.file_location = "new_data_dir"
        except ValueError as e:
            print(f"Caught expected error: {e}")

        # --- 6. Simulate re-loading the manager ---
        print("\n--- Re-initializing Manager (Loading from file) ---")
        # Create a new instance pointing to the *same file*
        manager_2 = JsonExperimentManager(
            file_location=test_dir, file_name="prod_results.json"
        )
        # It should load the 2 experiments we already saved
        print(f"Manager 2 loaded {len(manager_2.get_all_experiments())} experiments.")

        manager_2.add_experiment(
            arguments={"voltage": 5, "material": "silver"},
            data={"resistance": 0.8, "time_ms": 120},
        )
        # Now the file will contain 3 experiments
        print(
            f"Manager 2 now shows {len(manager_2.get_all_experiments())} total experiments."
        )

        print("\n--- Test Complete ---")

    except Exception as e:
        print(f"\nAn unexpected error occurred during the example run: {e}")
