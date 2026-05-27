import os
import warnings
import torch
import jax

# Check hardware availability and quarantine JAX to CPU if only 1 GPU exists
num_gpus_available = torch.cuda.device_count()
if num_gpus_available <= 1:
    jax.config.update("jax_platform_name", "cpu")
    print(
        f"Hardware Arbitration: {num_gpus_available} GPU(s) found. JAX forced to CPU."
    )
else:
    jax.config.update("jax_platform_name", "gpu")
    print(
        f"Hardware Arbitration: {num_gpus_available} GPUs found. JAX assigned to GPU 0."
    )

import numpy as np
from tqdm import tqdm, trange
import torch.multiprocessing as mp
import argparse

from models.configurableAutoencoder import Autoencoder

from models import autoencoderConfigs
from utils.mmd_test import mmd_test
from utils.energy_test import energy_test
from utils.bks import bks_distance_test
from utils.mmd_agg import mmdAgg_test
from data.data_builder import get_dataloader, get_seeded_random_dataloader
from data.data_logging import JsonExperimentManager, JsonStyle, JsonDict
from pathlib import Path

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


# ---------Feature extraction---------
def extract_features(model, loader, device):
    model.eval()
    feats = []

    is_parallel = isinstance(model, torch.nn.DataParallel)

    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)

            z = model(imgs, return_encoding=True)

            if z.dim() > 2:
                raise ValueError("Images are still in the pixel space")
                z = z.view(z.size(0), -1)

            # Safely move to CPU numpy array to prevent PyTorch/JAX memory collisions
            feats.append(z.cpu().numpy())

    return np.concatenate(feats, axis=0)


class ShiftExperiment:
    def __init__(
        self,
        source_dir: str = "./datasets/CULane",
        target_dir: str = "./datasets/CULane",
        source_list_path: str = "./datasets/CULane/list/train.txt",
        target_list_path: str = "./datasets/CULane/list/test.txt",
        sample_size: int = 1000,
        num_runs: int = 100,
        block_idx: int = 0,
        batch_size: int = 1024,
        image_size: int = 512,
        alpha: float = 0.05,
        seed_base: int = 42,
        file_name: str = "testData.json",
        file_location: str = "./",
        file_style: JsonStyle = 4,
        permutation_test_iterations: int = 1000,
        latent_dim: int = 32,
        model_weights_path: str = "",
        imagenet_weights: bool = False,
    ):
        # Assert well formed input
        if not imagenet_weights:
            assert (
                model_weights_path := Path(model_weights_path).resolve()
            ).is_dir(), (
                f"For arg model_weights_path: \n{model_weights_path}\nIs not a valid directory"
            )
        else:
            # keep a resolved placeholder path for logging consistency
            model_weights_path = str(Path(model_weights_path).resolve())

        assert (
            file_location := Path(file_location).resolve()
        ).is_dir(), (
            f"For arg file_location: \n{file_location}\nIs not a valid directory"
        )

        assert (
            source_dir := Path(source_dir).resolve()
        ).is_dir(), f"For arg source_dir: \n{source_dir}\nIs not a valid directory"

        assert (
            target_dir := Path(target_dir).resolve()
        ).is_dir(), (
            f"For arg target_dir: \n{target_dir}\nIs not a valid directory"
        )

        self.source_dir = source_dir
        self.target_dir = target_dir
        self.source_list_dir = source_list_path
        self.target_list_dir = target_list_path
        self.sample_size = sample_size
        self.num_runs = num_runs
        self.block_idx = block_idx
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_calib = num_runs
        self.alpha = alpha
        self.seed_base = seed_base
        self.permutation_test_iterations = permutation_test_iterations
        self.latent_dim = latent_dim
        self.weights_path = model_weights_path
        self.imagenet_weights = imagenet_weights

        self.test_types = ["MMD", "MMD_Agg", "Energy", "BKS"]

        # Data Storage for Preloaded Features
        self.src_feats = None
        self.calib_data_cache = []
        self.sanity_data_cache = {}
        self.target_data_cache = []

        # Null Stats & Thresholds Tracking
        self.null_stats = {}
        self.tau = {}

        self.datalogger = JsonExperimentManager(
            file_location=file_location, file_name=file_name, style=file_style
        )

        self.loggerArgs: JsonDict = {
            "CodeMark": "4/11/2026",
            "source_dir": str(source_dir),
            "target_dir": str(target_dir),
            "source_list_path": str(source_list_path),
            "target_list_path": str(target_list_path),
            "sample_size": sample_size,
            "num_runs": num_runs,
            "block_idx": block_idx,
            "batch_size": batch_size,
            "image_size": image_size,
            "tau_threshold_percentile": 100 * (1 - alpha),
            "seed_base": seed_base,
            "alpha": alpha,
            "deterministic": True,
            "permutation_test_iterations": permutation_test_iterations,
            "latent_dim": latent_dim,
            "test_types": self.test_types,
            "imagenet_weights": bool(imagenet_weights),
            "model_weights_path": str(model_weights_path),
        }

        self.loggerExperimentalData: JsonDict = {}

        # --- GPU Setup ---
        num_gpus = torch.cuda.device_count()

        if num_gpus <= 1:
            self.device = torch.device("cuda:0" if num_gpus == 1 else "cpu")
            self.torch_device_ids = [0] if num_gpus == 1 else None
        else:
            # Leave GPU 0 for JAX, use GPUs 1 to N for Torch
            self.torch_device_ids = list(range(1, num_gpus))
            self.device = torch.device(f"cuda:{self.torch_device_ids[0]}")

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(42)
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(42)

        # --- Model Initialization ---
        print("\nInitializing autoencoder...")

        # Define model parameters based on config
        if self.imagenet_weights:
            base_model = Autoencoder(
                latent_dim=self.latent_dim, imagenet_weights=True
            ).to(self.device)
        else:
            base_model = Autoencoder(
                latent_dim=self.latent_dim, weights_path=self.weights_path
            ).to(self.device)

        if num_gpus > 2:
            self.model = torch.nn.DataParallel(
                base_model, device_ids=self.torch_device_ids
            )
            print(f"PyTorch parallelized across GPUs: {self.torch_device_ids}")
        elif num_gpus == 2:
            self.model = base_model
            print("PyTorch assigned strictly to GPU 1.")
        elif num_gpus == 1:
            self.model = base_model
            print("PyTorch assigned to GPU 0. JAX running on CPU.")
        else:
            self.model = base_model
            print("Using CPU for everything.")

    # --- LATENT CACHING HELPER ---
    def _get_or_extract_features(self, loader, list_path, num_samples, seed):
        cache_base = "Encodings"
        cache_dir = os.path.join(cache_base, f"dim_{self.latent_dim}")

        os.makedirs(cache_dir, exist_ok=True)

        safe_list_name = (
            os.path.normpath(list_path).replace(os.sep, "_").replace(".", "_")
        )
        file_name = f"{safe_list_name}_n{num_samples}_seed{seed}.npy"
        file_path = os.path.join(cache_dir, file_name)

        if os.path.exists(file_path):
            return np.load(file_path)
        else:
            feats = extract_features(self.model, loader, self.device)
            np.save(file_path, feats)
            return feats

    # --- SEQUENTIAL WORKER FOR STATISTICAL TESTS ---
    def _execute_test(self, src_feats, tgt_feats, seed):
        mmd_results = mmd_test(
            src_feats, tgt_feats, iterations=self.permutation_test_iterations
        )
        mmdAgg_results = mmdAgg_test(
            src_feats, tgt_feats, iterations=self.permutation_test_iterations, seed=seed
        )
        energy_results = energy_test(
            src_feats, tgt_feats, iterations=self.permutation_test_iterations
        )
        bks_results = bks_distance_test(src_feats, tgt_feats)

        return {
            "MMD": mmd_results,
            "MMD_Agg": mmdAgg_results,
            "Energy": energy_results,
            "BKS": bks_results,
        }

    def preload_all_features(self):
        print("\n[STEP 0] Encoding and caching all dataset features...")

        loader_kwargs = {"num_workers": 14, "pin_memory": True, "prefetch_factor": 2}

        # 0A: Source Features
        loaderReturn = get_dataloader(
            root_dir=self.source_dir,
            list_path=self.source_list_dir,
            batch_size=self.batch_size,
            image_size=self.image_size,
            num_samples=self.sample_size,
        )

        self.src_feats = self._get_or_extract_features(
            loaderReturn[0], self.source_list_dir, self.sample_size, "base"
        )

        print(f"{self.source_dir} features loaded. Shape = {self.src_feats.shape}\n")
        self.loggerExperimentalData["Source Training Feature Shape"] = list(
            self.src_feats.shape
        )
        self.loggerExperimentalData["Source Training Feature Image Paths"] = list(
            loaderReturn[1]
        )

        # 0B: Calibration Features
        for i in tqdm(range(self.num_calib), desc="Preloading Calibration Sets"):
            seed = self.seed_base + i
            loaderReturn = get_seeded_random_dataloader(
                root_dir=self.source_dir,
                list_path=self.source_list_dir,
                batch_size=self.batch_size,
                image_size=self.image_size,
                num_samples=self.sample_size,
                seed=seed,
            )

            feats = self._get_or_extract_features(
                loaderReturn[0], self.source_list_dir, self.sample_size, seed
            )

            self.calib_data_cache.append((seed, feats, loaderReturn[1]))

        # 0C: Sanity Check Features
        sanity_seed = int(self.seed_base + self.num_calib)
        loaderReturn = get_seeded_random_dataloader(
            root_dir=self.source_dir,
            list_path=self.source_list_dir,
            batch_size=self.batch_size,
            image_size=self.image_size,
            num_samples=self.sample_size,
            seed=sanity_seed,
            shift=None,
        )
        sanity_feats = self._get_or_extract_features(
            loaderReturn[0], self.source_list_dir, self.sample_size, sanity_seed
        )
        self.sanity_data_cache = {
            "seed": sanity_seed,
            "feats": sanity_feats,
            "paths": loaderReturn[1],
        }
        print("-> Sanity Check features loaded.")

        # 0D: Target Data Shift Features
        for i in tqdm(range(self.num_runs), desc="Preloading Target Sets"):
            seed = self.seed_base + self.num_calib + i
            loaderReturn = get_seeded_random_dataloader(
                root_dir=self.target_dir,
                list_path=self.target_list_dir,
                batch_size=self.batch_size,
                image_size=self.image_size,
                num_samples=self.sample_size,
                seed=seed,
            )
            feats = self._get_or_extract_features(
                loaderReturn[0], self.target_list_dir, self.sample_size, seed
            )
            self.target_data_cache.append((seed, feats, loaderReturn[1]))

    # STEP 1 — Calibration (Null Distribution) via Sequential JAX
    def calibrate(self):
        calibrationData: JsonDict = {}
        print(f"\n[STEP 1] Calibration using {self.source_dir} (Sequential JAX)...")
        calibrationData["Uses"] = self.source_dir

        null_stats_temp = {test: [] for test in self.test_types}
        p_values_temp = {test: [] for test in self.test_types}
        all_image_dirs = {}

        # Standard Sequential Loop (No Threads)
        for seed, feats, img_paths in tqdm(
            self.calib_data_cache, desc="Calculating Calibration"
        ):
            all_image_dirs[f"Calibrating with seed {seed}"] = img_paths

            try:
                # JAX handles the parallelization on the GPU directly
                result_dict = self._execute_test(self.src_feats, feats, seed)

                for test in self.test_types:
                    t_stat, p_value = result_dict[test]
                    null_stats_temp[test].append(t_stat)
                    if self.permutation_test_iterations > 0:
                        p_values_temp[test].append(float(p_value))

            except Exception as exc:
                print(f"Calibration generated an exception for seed {seed}: {exc}")

        calibrationData["Result"] = {}

        # Calculate limits for each test
        for test in self.test_types:
            self.null_stats[test] = np.array(null_stats_temp[test])
            self.tau[test] = np.percentile(
                self.null_stats[test], 100 * (1 - self.alpha)
            )

            print(f"  [{test}] τ({1 - self.alpha:.2f}) = {self.tau[test]:.6f}")

            calibrationData["Result"][test] = {
                "Tau": float(self.tau[test]),
                "Mean": float(self.null_stats[test].mean()),
                "Std": float(self.null_stats[test].std()),
            }

            if self.permutation_test_iterations > 0 and len(p_values_temp[test]) > 0:
                avgPVal = np.array(p_values_temp[test]).mean()
                calibrationData["Result"][test]["Average P-Value"] = float(avgPVal)
                calibrationData["Result"][test]["P-Values"] = p_values_temp[test]

        self.loggerExperimentalData["Calibration"] = calibrationData

    # STEP 2 — Sanity Check
    def sanity_check(self):
        sanityCheckData: JsonDict = {}
        print("\n[STEP 2] Sanity Check...")

        sanityCheckData["Image Paths"] = self.sanity_data_cache["paths"]
        sanity_seed = self.sanity_data_cache["seed"]
        sanity_feats = self.sanity_data_cache["feats"]

        result_dict = self._execute_test(self.src_feats, sanity_feats, sanity_seed)
        sanityCheckData["Results"] = {}

        for test in self.test_types:
            t_stat, p_value = result_dict[test]

            print(
                f"[SANITY CHECK] {test}(A={self.source_dir}, B={self.source_dir}) = {t_stat:.6f}, τ = {self.tau[test]:.6f}"
            )

            test_data = {
                "Sanity Check Definition": f"{test}(A={self.source_dir}, B={self.source_dir})",
                "Stat": float(t_stat),
                "Tau": float(self.tau[test]),
            }

            if self.permutation_test_iterations > 0:
                test_data["P-Value"] = float(p_value)

            if t_stat <= self.tau[test]:
                test_data["Shift Detected"] = False
                print(f"No shift detected for {test}.")
            else:
                test_data["Shift Detected"] = True
                print(f"False shift detected for {test}.")
                warnings.warn(
                    f"False shift detected in sanity check - {test} exceeded threshold",
                    UserWarning,
                )

            sanityCheckData["Results"][test] = test_data

        print("")
        self.loggerExperimentalData["Sanity Check"] = sanityCheckData

    # STEP 3 — Data Shift Test via Sequential JAX
    def data_shift_test(self):
        dataShiftTestData: JsonDict = {}
        print(f"[STEP 3] Data Shift Test: {self.source_dir} to {self.target_dir}\n")

        dataShiftTestData["Data Shift Test Definition"] = (
            f"{self.source_dir} to {self.target_dir}"
        )
        dataShiftTestData["Runs"] = self.num_runs

        tpr_lists = {test: [] for test in self.test_types}
        stat_values = {test: [] for test in self.test_types}
        dataShiftTestDataTests: list[JsonDict] = []

        # Standard Sequential Loop (No Threads)
        for idx, (seed, feats, img_paths) in enumerate(
            tqdm(self.target_data_cache, desc="Calculating Shifts")
        ):
            runData: JsonDict = {
                "Image Paths": img_paths,
                "Seed": seed,
                "Run": int(idx + 1),
                "Results": {},
            }

            try:
                # JAX handles the parallelization on the GPU directly
                result_dict = self._execute_test(self.src_feats, feats, seed)

                for test in self.test_types:
                    t_stat, p_value = result_dict[test]
                    stat_values[test].append(t_stat)

                    detected: bool = t_stat > self.tau[test]
                    tpr_lists[test].append(int(detected))

                    test_run = {"Stat": float(t_stat), "Shift Detected": bool(detected)}

                    if self.permutation_test_iterations > 0:
                        test_run["P-Value"] = float(p_value)

                    runData["Results"][test] = test_run

                dataShiftTestDataTests.append(runData)

            except Exception as exc:
                print(f"Data shift test generated an exception for run {idx+1}: {exc}")

        dataShiftTestData["Individual Test Data"] = dataShiftTestDataTests
        dataShiftTestData["Summary"] = {}

        print("\n[RESULTS] Data Shift detection summary:")
        for test in self.test_types:
            tpr_result = np.mean(tpr_lists[test])
            mean_stat = np.mean(stat_values[test])
            std_stat = np.std(stat_values[test])

            print(f"--- {test} ---")
            print(f"  Average Stat: {mean_stat:.6f} ± {std_stat:.6f}")
            print(
                f"  TPR (true positive rate) over {self.num_runs} runs: {tpr_result*100:.2f}%"
            )

            dataShiftTestData["Summary"][test] = {
                "TPR": float(tpr_result * 100),
                "Mean Stat": float(mean_stat),
                "Std Stat": float(std_stat),
            }

        self.loggerExperimentalData["Data Shift Test Data"] = dataShiftTestData

    # RUN EVERYTHING
    def run(self):
        # Phase 1: Extract/Load all encodings safely (No Statistical Testing yet)
        self.preload_all_features()

        # Phase 2: Sequential Statistical Tests via JAX (No PyTorch interference)
        self.calibrate()
        self.sanity_check()
        self.data_shift_test()

        # Phase 3: Log Data
        self.datalogger.add_experiment(
            arguments=self.loggerArgs, data=self.loggerExperimentalData
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Common args shared by all modes
    parser.add_argument("--source_dir", required=True, type=str)
    parser.add_argument("--target_dir", required=True, type=str)
    parser.add_argument(
        "--source_list_path",
        type=str,
        default="./datasets/CULane/list/train.txt",
    )
    parser.add_argument("--target_list_path", required=True, type=str)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--block_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--permutation_test_iterations", type=int, default=1000)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument(
        "--file_location",
        type=str,
        default="logs",
        help="Directory to save the log file.",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="experiment.json",
        help="Name of the log file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Use custom weights directory
    custom_weights = subparsers.add_parser(
        "custom_weights", help="Run the experiment with custom weights."
    )
    custom_weights.add_argument(
        "--model_weights_path",
        type=str,
        required=True,
        help="Path to the DIRECTORY containing the custom weights.",
    )

    # Use ImageNet weights
    imagenet = subparsers.add_parser(
        "imagenet_weights", help="Run the experiment using ImageNet weights."
    )

    args = parser.parse_args()

    # Build kwargs that ShiftExperiment actually accepts
    exp_kwargs = dict(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        source_list_path=args.source_list_path,
        target_list_path=args.target_list_path,
        sample_size=args.sample_size,
        num_runs=args.num_runs,
        block_idx=args.block_idx,
        batch_size=args.batch_size,
        image_size=args.image_size,
        alpha=args.alpha,
        seed_base=args.seed_base,
        file_name=args.file_name,
        file_location=args.file_location,
        permutation_test_iterations=args.permutation_test_iterations,
        latent_dim=args.latent_dim,
    )

    if args.command == "custom_weights":
        exp_kwargs["imagenet_weights"] = False
        exp_kwargs["model_weights_path"] = args.model_weights_path
    elif args.command == "imagenet_weights":
        exp_kwargs["imagenet_weights"] = True
        exp_kwargs["model_weights_path"] = ""  # unused in this mode

    ShiftExperiment(**exp_kwargs).run()
