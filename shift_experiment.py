import os
import numpy as np
from tqdm import trange
import torch
from models.autoencoder import ConvAutoencoderFC
import argparse

from data.data_utils import (
    GaussianShift,
    RotationShift,
    TranslationShift,
    ShearShift,
    ZoomShift,
    HorizontalFlipShift,
    VerticalFlipShift,
)
from utils.mmd_test import mmd_test
from data.data_builder import get_dataloader, get_seeded_random_dataloader
from data.data_logging import JsonExperimentManager, JsonStyle, JsonDict


# ---------Feature extraction---------
def extract_features(model, loader, device):
    model.eval()
    feats = []

    is_parallel = isinstance(model, torch.nn.DataParallel)

    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)

            if is_parallel:
                z = model.module.encode(imgs)
            else:
                z = model.encode(imgs)
            
            if z.dim() > 2:
                raise ValueError("Images are still in the pixel space")
                z = z.view(
                    z.size(0), -1
                )  # code to run on raw images (to flatten the image and do the tests)

            feats.append(z.cpu().numpy())
    return np.concatenate(feats, axis=0)


class ShiftExperiment:
    def __init__(
        self,
        source_dir: str = "./datasets/CULane",
        target_dir: str = "./datasets/CULane",
        source_list_path: str = "./datasets/CULane/list/train.txt",
        source_test_list_path: str = "./datasets/CULane/list/test.txt",
        target_list_path: str = "./datasets/CULane/list/test.txt",
        src_samples: int = 1000,  # No. of source samples as train set passed
        tgt_samples: int = 100,
        num_runs: int = 10,
        block_idx: int = 0,  # block of samples selected from the the text file
        batch_size: int = 16,  # batch processing of data within an epoch
        image_size: int = 512,
        num_calib: int = 100,
        alpha: float = 0.05,
        seed_base: int = 42,
        shift: str = None,
        std: float = 0.0,
        cropImg: bool = False,
        rotation_angle: float = 45.0,
        width_shift_frac: float = 0.2,
        height_shift_frac: float = 0.2,
        shear_angle: float = 20.0,
        zoom_factor: float = 1.3,  # Guide: 1.3 is 30% zoom in and 0.7 is 30% zoom out
        file_name: str = "testData.json",
        file_location: str = "./",
        file_style: JsonStyle = 4,
        save_all_image_paths: bool = False,
    ):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.source_list_dir = source_list_path
        self.target_list_dir = target_list_path
        self.src_samples = src_samples
        self.tgt_samples = tgt_samples
        self.num_runs = num_runs
        self.block_idx = block_idx
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_calib = num_calib
        self.alpha = alpha
        self.seed_base = seed_base
        self.shift_type = shift
        self.std = std
        self.cropImg = cropImg
        self.rotation_angle = rotation_angle
        self.width_shift_frac = width_shift_frac
        self.height_shift_frac = height_shift_frac
        self.shear_angle = shear_angle
        self.zoom_factor = zoom_factor
        self.save_all_image_paths = save_all_image_paths

        # ------------------ Data Logger Config  ------------------

        self.datalogger = JsonExperimentManager(
            file_location=file_location, file_name=file_name, style=file_style
        )

        self.loggerArgs: JsonDict = {
            "source_dir": source_dir,
            "target_dir": target_dir,
            "source_list_path": source_list_path,
            "target_list_path": target_list_path,
            "src_samples": src_samples,
            "tgt_samples": tgt_samples,
            "num_runs": num_runs,
            "block_idx": block_idx,
            "batch_size": batch_size,
            "image_size": image_size,
            "num_calib": num_calib,
            "alpha": alpha,
            "seed_base": seed_base,
            "shift": shift,
            "std": std,
            "cropImg": cropImg,
            "rotation_angle": rotation_angle,
            "width_shift_frac": width_shift_frac,
            "height_shift_frac": height_shift_frac,
            "shear_angle": shear_angle,
            "zoom_factor": zoom_factor,
        }

        self.loggerExperimentalData: JsonDict = {}

        # ------------------ Check for GPU ------------------

        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        self.loggerExperimentalData["CUDA"] = cuda_available

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs("features", exist_ok=True)

        # ------------------ Model ------------------
        print("\nInitializing autoencoder...")
        self.model = ConvAutoencoderFC(latent_dim=512, pretrained=True).to(self.device)

        # Initialialize shift object
        self.shift_object = None
        if self.shift_type == "gaussian":
            if self.std == 0.0:
                raise ValueError(
                    "Gaussian noise selected but std=0. Please provide > 0 --std value."
                )
            self.shift_object = GaussianShift(std=self.std)
        elif self.shift_type == "rotation_shift":
            if self.rotation_angle == 0.0:
                raise ValueError(
                    "Rotation angle selected = 0. Please provide valid --rotation_angle value."
                )
            self.shift_object = RotationShift(angle=self.rotation_angle)
        elif self.shift_type == "translation_shift":
            if self.width_shift_frac == 0.0 or self.height_shift_frac == 0:
                raise ValueError(
                    "width_shift_frac or height_shift_frac angle selected = 0. Please provide valid --width_shift_frac or --height_shift_frac value."
                )
            self.shift_object = TranslationShift(
                width_shift_frac=self.width_shift_frac,
                height_shift_frac=self.height_shift_frac,
            )
        elif self.shift_type == "shear_shift":
            if self.shear_angle == 0.0:
                raise ValueError(
                    "Shear angle selected = 0. Please provide valid --shear_angle value."
                )
            self.shift_object = ShearShift(shear_angle=self.shear_angle)
        elif self.shift_type == "zoom_shift":
            if self.zoom_factor == 1.0:
                raise ValueError(
                    "Zoom factor selected = 1. Please provide valid --zoom_factor value."
                )
            self.shift_object = ZoomShift(zoom_factor=self.zoom_factor)
        elif self.shift_type == "horizontal_flip_shift":
            self.shift_object = HorizontalFlipShift()
        elif self.shift_type == "vertical_flip_shift":
            self.shift_object = VerticalFlipShift()

    # STEP 0 — Load Source Features
    def load_source_features(self):
        loaderReturn = get_dataloader(
            root_dir=self.source_dir,
            list_path=self.source_list_dir,
            batch_size=self.batch_size,
            image_size=self.image_size,
            num_samples=self.src_samples,
            cropImg=self.cropImg,
            block_idx=self.block_idx,
        )
        loader = loaderReturn[0]
        image_paths = loaderReturn[1]
        self.src_feats = extract_features(self.model, loader, self.device)
        print(f"{self.source_dir} features loaded. Shape = {self.src_feats.shape}\n")
        self.loggerExperimentalData["Source Features Shape"] = list(
            self.src_feats.shape
        )
        self.loggerExperimentalData["Source Features Image Paths"] = list(image_paths)

    # STEP 1 — Calibration (Null Distribution)
    def calibrate(self):
        calibrationData: JsonDict = {}
        print(f"[STEP 1] Calibration using {self.source_dir}...")
        calibrationData["Uses"] = self.source_dir
        null_stats = []
        all_image_dirs = {}

        for i in trange(self.num_calib, desc="Calibrating"):
            seed = self.seed_base + i
            dataloaderReturn = get_seeded_random_dataloader(
                root_dir=self.source_dir,
                list_path=self.source_list_dir,
                batch_size=self.batch_size,
                image_size=self.image_size,
                num_samples=self.tgt_samples,
                seed=seed,
                cropImg=self.cropImg,
                shift=None,
            )
            calib_src_loader = dataloaderReturn[0]
            all_image_dirs[f"Calibrating with seed {seed}"] = dataloaderReturn[1]
            calib_src_feats = extract_features(
                self.model, calib_src_loader, self.device
            )

            t_stat = mmd_test(self.src_feats, calib_src_feats)
            null_stats.append(t_stat)

        self.null_stats = np.array(null_stats)
        self.tau = np.percentile(self.null_stats, 100 * (1 - self.alpha))

        print(f"\n[RESULT] τ({1 - self.alpha:.2f}) = {self.tau:.6f}")
        print(
            f"Mean MMD (same-distribution): {self.null_stats.mean():.6f} ± {self.null_stats.std():.6f}\n"
        )
        calibrationData["Result"] = {
            "Tau": float(self.tau),
            "Mean MMD": float(self.null_stats.mean()),
            "MMD (std)": float(self.null_stats.std()),
        }
        self.loggerExperimentalData["Calibration"] = calibrationData

    # STEP 2 — Sanity Check
    def sanity_check(self):
        sanityCheckData: JsonDict = {}
        print("[STEP 2] Sanity Check...")

        loaderReturn = get_seeded_random_dataloader(
            root_dir=self.source_dir,
            list_path=self.source_list_dir,
            batch_size=self.batch_size,
            image_size=self.image_size,
            num_samples=self.tgt_samples,
            seed=int(self.seed_base + 1),
            cropImg=self.cropImg,
            shift=None,
        )
        sanity_src_loader = loaderReturn[0]
        sanityCheckData["Image Paths"] = loaderReturn[1]

        sanity_src_feats = extract_features(self.model, sanity_src_loader, self.device)

        mmd_val = mmd_test(self.src_feats, sanity_src_feats)
        print(
            f"[SANITY CHECK] MMD({self.source_dir}to{self.source_dir}) = {mmd_val:.6f}, τ = {self.tau:.6f}"
        )
        sanityCheckData["Results"] = {
            "Sanity Check Definition": f"{self.source_dir}to{self.source_dir}",
            "MMD": float(mmd_val),
            "Tau": float(self.tau),
        }

        if mmd_val <= self.tau:
            sanityCheckData["Shift Detected"] = bool(False)
            print("No shift detected.\n")
        else:
            sanityCheckData["Shift Detected"] = bool(True)
            print("False shift detected.\n")

        self.loggerExperimentalData["Sanity Check"] = sanityCheckData

    # STEP 3 — Data Shift Test
    def data_shift_test(self):
        dataShiftTestData: JsonDict = {}
        print(
            f"[STEP 3] Data Shift Test: {self.source_dir} to {self.target_dir}, Noise applied: {self.shift_object}\n"
        )
        dataShiftTestData["Data Shift Test Definition"] = (
            f"{self.source_dir} to {self.target_dir}"
        )
        dataShiftTestData["Noise Applied"] = str(self.shift_object)
        dataShiftTestData["Runs"] = self.num_runs

        tpr_list = []
        mmd_values = []
        dataShiftTestDataTests: list[JsonDict] = []

        for i in trange(self.num_runs, desc="Shift Testing"):
            testData: JsonDict = {}
            seed = self.seed_base + i
            loaderReturn = get_seeded_random_dataloader(
                root_dir=self.target_dir,
                list_path=self.target_list_dir,
                batch_size=self.batch_size,
                image_size=self.image_size,
                num_samples=self.tgt_samples,
                seed=seed,
                cropImg=self.cropImg,
                shift=self.shift_object,
            )
            tgt_loader_cross = loaderReturn[0]
            testData["Image Paths"] = loaderReturn[1]
            testData["Seed"] = seed
            tgt_feats_cross = extract_features(
                self.model, tgt_loader_cross, self.device
            )
            mmd_cross = mmd_test(self.src_feats, tgt_feats_cross)

            mmd_values.append(mmd_cross)
            detected: bool = mmd_cross > self.tau
            tpr_list.append(int(detected))

            print(f"[RUN {i+1}] MMD={mmd_cross:.6f} {'✅' if detected else '❌'}")
            testData["Run"] = int(i + 1)
            testData["MMD"] = float(mmd_cross)
            testData["Shift Detected"] = bool(detected)

            if self.save_all_image_paths:
                dataShiftTestDataTests.append(testData)
            elif i == 0:
                dataShiftTestDataTests.append(testData)

        dataShiftTestData["Individual Test Data"] = dataShiftTestDataTests

        tpr_result = np.mean(tpr_list)
        print("\n[RESULTS] Data Shift detection summary")
        print(f"Noise Applied: {self.shift_object}")
        print(f"Average MMD: {np.mean(mmd_values):.6f} ± {np.std(mmd_values):.6f}")
        print(
            f"TPR (true positive rate) over {self.num_runs} runs: {tpr_result*100:.2f}%"
        )
        dataShiftTestData["TPR"] = float(tpr_result * 100)
        dataShiftTestData["Average MMD"] = float(np.mean(mmd_values))
        dataShiftTestData["Average MMD (std)"] = float(np.std(mmd_values))
        self.loggerExperimentalData["Data Shift Test Data"] = dataShiftTestData

    # RUN EVERYTHING
    def run(self):
        # Step 0
        self.load_source_features()
        # Step 1
        self.calibrate()
        # Step 2
        self.sanity_check()
        # Step 3
        self.data_shift_test()
        # Log Data
        self.datalogger.add_experiment(
            arguments=self.loggerArgs, data=self.loggerExperimentalData
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_dir", required=True, type=str, default="./datasets/CULane"
    )
    parser.add_argument(
        "--target_dir", required=True, type=str, default="./datasets/Curvelanes"
    )
    parser.add_argument(
        "--source_list_path",
        required=True,
        type=str,
        default="./datasets/CULane/list/train.txt",
    )
    parser.add_argument(
        "--target_list_path",
        required=True,
        type=str,
        default="./datasets/Curvelanes/train/train.txt",
    )
    parser.add_argument("--src_samples", type=int, default=1000)
    parser.add_argument("--tgt_samples", type=int, default=100)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--block_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_calib", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--shift", type=str, default=None)
    parser.add_argument("--std", type=float, default=0.0)
    parser.add_argument("--cropImg", type=bool, default=False)
    parser.add_argument("--rotation_angle", type=float, default=0.0)
    parser.add_argument("--shear_angle", type=float, default=0.0)
    parser.add_argument("--zoom_factor", type=float, default=1.0)
    parser.add_argument("--width_shift_frac", type=float, default=0.2)
    parser.add_argument("--height_shift_frac", type=float, default=0.2)
    parser.add_argument("--save_all_image_paths", type=bool, default=False)
    parser.add_argument(
        "--file_location",
        type=str,
        default="logs",
        help="Directory to save the log file.",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="sanity_check.json",
        help="Name of the log file.",
    )

    args = parser.parse_args()

    ShiftExperiment(**vars(args)).run()