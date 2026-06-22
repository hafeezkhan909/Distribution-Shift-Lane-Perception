import argparse
import glob
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from configurableAutoencoder import Autoencoder
from data.data_builder import get_dataloader

# Configure logging style for clean terminal reporting
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Data Loading ---
    logging.info(f"Loading dataset: {args.dataset_name}...")
    train_loader, _ = get_dataloader(
        root_dir=args.dataset_dir,
        list_path=args.dataset_list,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_samples=args.samples,
        cropImg=args.cropImg,
        block_idx=args.block_idx,
    )
    logging.info(f"Total training batches per epoch: {len(train_loader)}")

    # --- Resume Verification & Path Prep ---
    save_dir = f"checkpoints/Phase2/{args.dataset_name}"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_files = glob.glob(
        f"{save_dir}/P2autoencoder_{args.dataset_name}_epoch_*.pth"
    )
    latest_checkpoint = (
        max(checkpoint_files, key=os.path.getctime) if checkpoint_files else None
    )

    # --- Model Initialization ---
    # Handle constraints safely: if resuming, fall back to initializing with imagenet weights
    # to pass model assertions before safely overwriting the complete architecture weights.
    init_imagenet = args.imagenet_weights
    init_weights_path = args.weights_path

    if latest_checkpoint and not init_imagenet and not init_weights_path:
        logging.info(
            "Resume target detected. Overriding initialization constraints to fulfill checkpoint structure."
        )
        init_imagenet = True

    logging.info("Initializing Configurable Autoencoder...")
    model = Autoencoder(
        latent_dim=args.latent_dim,
        image_net=init_imagenet,
        weights_path=init_weights_path,
    ).to(device)

    # Multi-GPU setups
    if torch.cuda.device_count() > 1:
        logging.info(
            f"Using {torch.cuda.device_count()} GPUs for training via DataParallel."
        )
        model = nn.DataParallel(model)

    # --- Optimizer & Loss Function ---
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    mse_loss = nn.MSELoss()

    # --- Apply Resumed State If Available ---
    start_epoch = 0
    if latest_checkpoint:
        logging.info(
            f"Resume context detected! Loading checkpoint state: {latest_checkpoint}"
        )
        checkpoint = torch.load(latest_checkpoint, map_location=device)

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logging.info(
            f"Successfully loaded and resuming training from Epoch {start_epoch + 1}"
        )
    else:
        logging.info(
            "No prior checkpoint found. Starting fresh training session from Epoch 1."
        )

    # --- Core Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_recon_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
        )

        for imgs in progress_bar:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad()

            # Forward Pass
            output = model(imgs)

            # Defensive unwrap in case architecture mutations alter output format
            reconstructed = output[0] if isinstance(output, tuple) else output

            # Evaluation & Optimization Step
            loss = mse_loss(reconstructed, imgs)
            loss.backward()
            optimizer.step()

            # Tracking Metrics
            epoch_recon_loss += loss.item()
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        avg_loss = epoch_recon_loss / len(train_loader)
        logging.info(
            f"Epoch {epoch + 1} Complete -> Average Reconstruction Loss: {avg_loss:.5f}"
        )

        # --- Checkpoint Saving ---
        checkpoint_path = (
            f"{save_dir}/P2autoencoder_{args.dataset_name}_epoch_{epoch + 1}.pth"
        )
        raw_model_state = (
            model.module.state_dict()
            if isinstance(model, nn.DataParallel)
            else model.state_dict()
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": raw_model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            },
            checkpoint_path,
        )
        logging.info(f"Checkpoint successfully stored at: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Publishing-grade Training Pipeline for Phase 2 Configurable Autoencoders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset Settings Group
    data_group = parser.add_argument_group("Dataset Options")
    data_group.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Target identity label of your dataset.",
    )
    data_group.add_argument(
        "--dataset_dir", type=str, required=True, help="Path pointing to raw data root."
    )
    data_group.add_argument(
        "--dataset_list",
        type=str,
        required=True,
        help="Path pointing to training list splits.",
    )
    data_group.add_argument(
        "--cropImg",
        action="store_true",
        help="When flagged, drops the bottom half of incoming matrices.",
    )
    data_group.add_argument(
        "--block_idx",
        type=int,
        default=0,
        help="Identifies which data block sequence to read.",
    )

    # Model Parameters Group
    model_group = parser.add_argument_group("Model Architecture Configurations")
    model_group.add_argument(
        "--latent_dim",
        type=int,
        default=32,
        help="Dimensional depth bottleneck vector size.",
    )
    model_group.add_argument(
        "--imagenet_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle usage of standard ImageNet configurations on Backbone initialization.",
    )
    model_group.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="Explicit target file path to baseline state dict weights.",
    )

    # General Hyperparameters Group
    hyper_group = parser.add_argument_group("Training Hyperparameters")
    hyper_group.add_argument(
        "--samples",
        type=int,
        default=100000,
        help="Hard ceiling sample constraint limit.",
    )
    hyper_group.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch dimensionality scale allocation per GPU step.",
    )
    hyper_group.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Dimension bounding box constraint for images.",
    )
    hyper_group.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum iteration loop execution cycles.",
    )
    hyper_group.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Starting optimizer scalar scaling rate.",
    )
    hyper_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random integer anchor used to define pipeline state repeatability.",
    )

    parsed_args = parser.parse_args()
    train(parsed_args)
