# gradcam_unet.py
import os
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2
from glob import glob

from wrappers import UNetSegWrapper
from models import UNet
from seg_xres_cam import seg_grad_cam

# -----------------
# Config
# -----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "./models/unet_lane_detection_best.pth"
image_dir = "./images/100k/val"
mask_dir = "./labels/100k/val"

# Output directories
base_outdir = "../gradcam"
os.makedirs(os.path.join(base_outdir, "method0"), exist_ok=True)
os.makedirs(os.path.join(base_outdir, "method1"), exist_ok=True)

# -----------------
# Load model
# -----------------
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
wrapped_model = UNetSegWrapper(model)

# -----------------
# Pick target layer
# -----------------
target_layer = model.bottleneck[-1]

# -----------------
# Transforms
# -----------------
transform = T.Compose(
    [
        T.Resize((256, 256)),
        T.ToTensor(),
    ]
)

resize_out = (512, 512)  # for final saved visualizations


# -----------------
# IoU utility
# -----------------
def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


# -----------------
# Iterate over 20 images
# -----------------
image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))[:20]

for img_path in image_paths:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(mask_dir, f"{base_name}.png")

    # load image + preprocess
    image_pil = Image.open(img_path).convert("RGB")
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    np_image = np.array(image_pil.resize((256, 256)))[:, :, ::-1]  # BGR for cv2

    # load GT mask
    if not os.path.exists(mask_path):
        print(f"Mask not found for {base_name}, skipping...")
        continue
    gt_mask = Image.open(mask_path).convert("L").resize((256, 256))
    gt_mask = (np.array(gt_mask) > 0).astype(np.uint8)

    # overlay GT mask in green
    gt_overlay = np_image.copy()
    gt_overlay[gt_mask == 1] = [0, 255, 0]

    # -----------------
    # Model prediction
    # -----------------
    with torch.no_grad():
        pred_logits = model(image_tensor)  # [1,1,H,W]
        pred_mask = torch.sigmoid(pred_logits).squeeze().cpu().numpy()
    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)

    # IoU
    iou = compute_iou(pred_mask_bin, gt_mask)
    print(f"{base_name} - IoU: {iou:.4f}")

    # Prediction visualization
    pred_overlay = np_image.copy()
    pred_overlay[pred_mask_bin == 1] = [255, 0, 0]  # red overlay for pred

    # -----------------
    # Run both CAM methods
    # -----------------
    for method_index in [0, 1]:
        grayscale_cam, overlaid = seg_grad_cam(
            image=transform(image_pil),
            model=wrapped_model,
            preprocess_transform=None,
            target=None,
            target_layer=target_layer,
            DEVICE=device,
            method_index=method_index,
            vis=False,
        )

        outdir = os.path.join(base_outdir, f"method{method_index}")
        os.makedirs(outdir, exist_ok=True)

        # Resize for saving
        np_img_resized = cv2.resize(np_image, resize_out)
        overlaid_bgr = cv2.resize(overlaid[:, :, ::-1], resize_out)  # RGB->BGR
        heatmap = cv2.resize((grayscale_cam * 255).astype(np.uint8), resize_out)
        gt_overlay_resized = cv2.resize(gt_overlay, resize_out)
        pred_overlay_resized = cv2.resize(pred_overlay, resize_out)
        pred_mask_resized = cv2.resize(pred_mask_bin * 255, resize_out)

        # Save
        cv2.imwrite(os.path.join(outdir, f"{base_name}.jpg"), np_img_resized)
        cv2.imwrite(os.path.join(outdir, f"{base_name}_overlay.jpg"), overlaid_bgr)
        cv2.imwrite(os.path.join(outdir, f"{base_name}_mask_overlay.jpg"), heatmap)
        cv2.imwrite(
            os.path.join(outdir, f"{base_name}_gt_overlay.jpg"), gt_overlay_resized
        )
        cv2.imwrite(
            os.path.join(outdir, f"{base_name}_pred_overlay.jpg"), pred_overlay_resized
        )
        cv2.imwrite(
            os.path.join(outdir, f"{base_name}_pred_mask.jpg"), pred_mask_resized
        )

print(
    "Saved Grad-CAM, XRes-CAM, GT overlays, predictions, and IoU scores for 20 validation images."
)
