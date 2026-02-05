import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConfConvAutoencoderFC32(nn.Module):
    def __init__(self, config="", pretrained=True):
        super().__init__()

        print(f"[ResNet-AE] dimensionallity: 32")

        # -------- Pretrained ResNet encoder --------
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        layers = list(backbone.children())[
            :-2
        ]  # Remove avgpool and fc layers (keep convs)
        self.encoder_conv = nn.Sequential(*layers)  # Output: (B, 512, H/32, W/32)

        # Freeze BatchNorm stats if using pretrained weights (UAE setting)
        for m in self.encoder_conv.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.requires_grad_(False)

        self.flatten_dim = 512 * 16 * 16  # for 512×512 input (ResNet downscales by /32)

        # -------- Fully Connected Encoder --------
        match config:
            case "d128rel":
                self.fc_encoder = nn.Sequential(
                    nn.Linear(self.flatten_dim, 4096),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 128)
                )
            case "d64rel":
                self.fc_encoder = nn.Sequential(
                    nn.Linear(self.flatten_dim, 4096),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 64)
                )
            case "d32rel":
                self.fc_encoder = nn.Sequential(
                    nn.Linear(self.flatten_dim, 4096),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 32)
                )
            case "d128gdd":
                self.fc_encoder = nn.Sequential(
                    nn.Linear(self.flatten_dim, 4096),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 128),
                )
            case "d64gdd":
                self.fc_encoder = nn.Sequential(
                    nn.Linear(self.flatten_dim, 4096),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 64),
                )
            case "d32gdd":
                self.fc_encoder = nn.Sequential(
                    nn.Linear(self.flatten_dim, 4096),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 32),
                )
            case "d64ids":
                self.fc_encoder = nn.Sequential(
                    nn.Linear(self.flatten_dim, 4096),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 64)
                )
            case "d128ids":
                self.fc_encoder = nn.Sequential(
                    nn.Linear(self.flatten_dim, 4096),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 128)
                )
            case "d32":
                self.fc_encoder = nn.Sequential(
                    nn.Linear(self.flatten_dim, 4096),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 32)
                )
            case "orig" | "":
                self.fc_encoder = nn.Sequential(
                    nn.Linear(self.flatten_dim, 4096),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 256),
                )
            case _:
                raise ValueError(f"Invalid config value: {config}.")

        self.fc_decoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.flatten_dim),
            nn.ReLU(inplace=True),
        )

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

    def encode(self, x):
        """Encode image → latent vector"""
        h = self.encoder_conv(x)  # (B, 512, 16, 16)
        h_flat = h.view(h.size(0), -1)
        z = self.fc_encoder(h_flat)
        return z

    def decode(self, z):
        """Decode latent vector → reconstructed image"""
        h_flat = self.fc_decoder(z)
        h = h_flat.view(z.size(0), 512, 16, 16)
        out = self.decoder_conv(h)
        return out

    def forward(self, x, return_encoding=False):
        z = self.encode(x)
        if return_encoding:
            return z
        out = self.decode(z)
        return out, z


if __name__ == "__main__":
    model = ResNetAutoencoder(latent_dim=512, pretrained=True)
    x = torch.randn(2, 3, 512, 512)
    out, z = model(x)
    print(f"[ResNet-AE] input: {x.shape}, latent: {z.shape}, recon: {out.shape}")
