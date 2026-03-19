import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

from models.phase2Autoencoder import ConfP2ConvAutoencoderFC 
from data.data_builder import get_dataloader 

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print(f"Loading dataset: {args.dataset_name}...")
    
    # Unpacking the dataloader and ignoring the returned image paths list
    train_loader, _ = get_dataloader(
        root_dir=args.dataset_dir, 
        list_path=args.dataset_list, 
        batch_size=args.batch_size, 
        image_size=args.image_size, 
        num_samples=args.samples,
        cropImg=args.cropImg,
        block_idx=args.block_idx
    )
    
    print(f"Total training batches per epoch: {len(train_loader)}")

    # --- Model Initialization ---
    model = ConfP2ConvAutoencoderFC().to(device)
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = nn.DataParallel(model)

    # --- Optimizer & Loss ---
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    mse_loss = nn.MSELoss()

    # --- Resume from Checkpoint Logic ---
    save_dir = f"checkpoints/Phase2/{args.dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    start_epoch = 0

    checkpoint_files = glob.glob(f"{save_dir}/P2autoencoder_{args.dataset_name}_epoch_*.pth")
    
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"Resume detected! Loading checkpoint: {latest_checkpoint}")
        
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from Epoch {start_epoch+1}")
    else:
        print("No prior checkpoint found. Starting fresh from Epoch 1.")

    # --- Training Loop ---
    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_recon_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for imgs in progress_bar:
            # Your Dataset.__getitem__ returns a raw tensor, so no need to unpack tuples
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad()

            # Forward pass
            output = model(imgs) 
            
            # If the model returns (reconstructed, z), just grab the reconstructed image
            reconstructed = output[0] if isinstance(output, tuple) else output
            
            # Reconstruction Loss
            loss = mse_loss(reconstructed, imgs)

            # Backprop
            loss.backward()
            optimizer.step()

            # Logging
            epoch_recon_loss += loss.item()
            progress_bar.set_postfix({"Recon Loss": f"{loss.item():.4f}"})

        avg_loss = epoch_recon_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss -> {avg_loss:.4f}")

        # Save checkpoint securely
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, f"{save_dir}/P2autoencoder_{args.dataset_name}_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Generic Dataset Paths
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--dataset_list", type=str, required=True)
    
    # Dataloader-specific args
    parser.add_argument("--cropImg", action="store_true", help="Include to crop the bottom half of images")
    parser.add_argument("--block_idx", type=int, default=0, help="Which chunk of the dataset to load")
    
    # Training Hyperparameters
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    args = parser.parse_args()
    train(args)
