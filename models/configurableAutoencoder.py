import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConfP2ConvAutoencoderFC(nn.Module):
    def __init__(
        self, latent_dim=32, weights_path=None
    ):
        super().__init__()

        print(f"[Autoencoder] - Latent Dim: {latent_dim}")

        # We will store the path here and load it AT THE END
        checkpoint_path = None

        # -------- Pretrained ResNet encoder --------
        if configs == "image_net":
            # Load ImageNet pretrained weights (UAE setting)
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif configs == "random_weights":
            # Load random weights (untrained ResNet)
            backbone = models.resnet18()
        elif configs == "cu_lane":
            # Load empty backbone, will populate at the end
            backbone = models.resnet18()
            checkpoint_path = "checkpoints/Phase2/CULane/P2autoencoder_CULane_epoch_50.pth"
        elif configs == "curvelanes":
            # Load empty backbone, will populate at the end
            backbone = models.resnet18()
            checkpoint_path = "checkpoints/Phase2/Curvelanes/P2autoencoder_Curvelanes_epoch_50.pth"
        elif configs == "assist_taxi":
            # Load empty backbone, will populate at the end
            backbone = models.resnet18()
            checkpoint_path = "checkpoints/Phase2/AssistTaxi/P2autoencoder_AssistTaxi_epoch_50.pth"
        else:
            raise ValueError(f"Unsupported config: {configs}")

        layers = list(backbone.children())[
            :-2
        ]  # Remove avgpool and fc layers (keep convs)
        self.encoder_conv = nn.Sequential(*layers)  # Output: (B, 512, H/32, W/32)

        # Freeze BatchNorm stats if using pretrained weights (UAE setting)
        for m in self.encoder_conv.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.requires_grad_(False)

        # New: Spatial Reduction Layer
        # 16x16 -> 4x4 (Factor of 16 reduction)
        self.spatial_pool = nn.AvgPool2d(kernel_size=4, stride=4)

        self.flatten_dim = 512 * 4 * 4  # 8192

        # -------- Fully Connected Encoder (2 Layers) --------
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),  # Latent dim = 32
        )

        # -------- Fully Connected Decoder (Symmetric) --------
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.flatten_dim),
            nn.ReLU(inplace=True),
        )

        # New: Upsample back to 16x16 before ConvTranspose starts
        self.spatial_upsample = nn.Upsample(scale_factor=4, mode="nearest")

        # -------- Decoder (ConvTranspose) --------
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 256x32x32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 128x64x64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64x128x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x256x256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # 3x512x512
            nn.Sigmoid(),
        )

        # -------- LOAD WEIGHTS AT THE END --------
        if checkpoint_path is not None:
            # map_location="cpu" ensures it loads safely into RAM before moving to GPU
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Successfully loaded full Autoencoder weights from {checkpoint_path}"
            )

    def encode(self, x):
        h = self.encoder_conv(x)  # (B, 512, 16, 16)
        h = self.spatial_pool(h)  # (B, 512, 4, 4)
        h_flat = h.reshape(h.size(0), -1)  # (B, 8192)
        z = self.fc_encoder(h_flat)  # (B, 32)
        return z

    def decode(self, z):
        h_flat = self.fc_decoder(z)  # (B, 8192)
        h = h_flat.view(z.size(0), 512, 4, 4)
        h = self.spatial_upsample(h)  # (B, 512, 16, 16)
        out = self.decoder_conv(h)  # (B, 3, 512, 512)
        return out

    def forward(self, x, return_encoding=False):
        # 1. Pass through encoder
        z = self.encode(x)

        # 2. Stop early if we just want the features
        if return_encoding:
            return z

        # 3. Otherwise, finish the forward pass (decoder)
        out = self.decode(z)
        return out
