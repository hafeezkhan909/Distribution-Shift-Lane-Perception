import os
import json
import glob
import argparse


def process_json_file(json_path, base_path=None):
    filename = os.path.basename(json_path)

    # The directory name for this JSON file
    json_dir_name = os.path.splitext(filename)[0]

    # If base_path is provided, output goes under that directory; otherwise alongside the file
    out_base = base_path if base_path is not None else os.path.dirname(json_path)
    json_dir_path = os.path.join(out_base, json_dir_name)

    # Load the JSON
    try:
        with open(json_path, "r") as f:
            schema = json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return

    experiments = schema.get("experiments", [])
    if not experiments:
        return

    print(f"Processing {filename} -> found {len(experiments)} experiments.")

    # Create the top-level directory for the JSON file
    os.makedirs(json_dir_path, exist_ok=True)

    # Iterate through each experiment
    for exp_idx, experiment in enumerate(experiments):
        # Get arguments for experiment type
        arg_block = experiment.get("arguments", {})

        # Use .get() with a default empty string just in case source/target are missing
        source_raw = arg_block.get("source_list_path", "")
        target_raw = arg_block.get("target_list_path", "")

        # 1-based indexing for folder names (Experiment_1, Experiment_2, etc.)
        exp_dir_path = os.path.join(
            json_dir_path, f"{source_raw}_to_{target_raw}_{exp_idx + 1}"
        )
        os.makedirs(exp_dir_path, exist_ok=True)

        dynamic_prefixes = []

        def get_dataset_root(txt_path):
            if not txt_path:
                return ""
            # Goes from ".../CULane/list/train.txt" -> ".../CULane/"
            root = os.path.dirname(os.path.dirname(txt_path))
            return root + "/" if not root.endswith("/") else root

        source_root = get_dataset_root(source_raw)
        target_root = get_dataset_root(target_raw)

        if source_root and source_root not in dynamic_prefixes:
            dynamic_prefixes.append(source_root)
        if target_root and target_root not in dynamic_prefixes:
            dynamic_prefixes.append(target_root)

        # Use a set to store unique paths for the Concat file
        concat_paths = set()

        data_block = experiment.get("data", {})
        for test_name, test_data in data_block.items():

            # If test_data isn't a dictionary, it can't contain "Individual Test Data". Skip it.
            if not isinstance(test_data, dict):
                continue

            individual_tests = test_data.get("Individual Test Data", [])

            # Iterate through each run
            for run in individual_tests:

                # Make sure the run is actually a dictionary before calling .get()
                if not isinstance(run, dict):
                    continue

                run_id = run.get("Run", "Unknown")
                paths = run.get("Image Paths", [])

                if not paths:
                    continue

                clean_paths = []
                for p in paths:
                    # Clean paths using the true dataset roots
                    for prefix in dynamic_prefixes:
                        if p.startswith(prefix):
                            p = p.replace(prefix, "", 1)
                            break  # Stop checking after the first matching prefix is found

                    # Remove any leftover leading slash so it's a true relative path
                    p = p.lstrip("/")
                    clean_paths.append(p)

                # 1. Write the individual run list file
                run_file_path = os.path.join(exp_dir_path, f"Run_{run_id}.txt")
                with open(run_file_path, "w") as f:
                    for p in clean_paths:
                        f.write(f"{p}\n")
                        concat_paths.add(p)  # Add to the master set for this experiment

        # 2. Write the concatenated list file for the experiment
        if concat_paths:
            concat_file_path = os.path.join(exp_dir_path, "full.txt")
            with open(concat_file_path, "w") as f:
                for p in sorted(concat_paths):
                    f.write(f"{p}\n")


def process_directory(base_path):
    # Find all json files in the target directory
    search_pattern = os.path.join(base_path, "*.json")
    json_files = glob.glob(search_pattern)

    if not json_files:
        print(f"No JSON files found in {base_path}")
        return

    for json_path in json_files:
        process_json_file(json_path, base_path=base_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract image lists into a nested directory structure."
    )
    parser.add_argument(
        "--dir",
        default=".",
        help="Directory containing the JSON files (default: current directory).",
    )
    parser.add_argument(
        "--file",
        help="Process a single JSON file and write outputs alongside it (or under --dir if provided).",
    )

    args = parser.parse_args()

    if args.file:
        # If both are provided, --file wins, but outputs under --dir if set explicitly
        out_base = args.dir if args.dir is not None else None
        process_json_file(args.file, base_path=out_base)
    else:
        process_directory(args.dir)
