"""
Grad-CAM visualizations.
Usage: python src/gradcam.py --experiment baseline
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model import FoodResNet
from dataset import get_transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="baseline")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = {}
    with open(os.path.join(args.data_dir, "class_list_food.txt"), "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                class_names[int(parts[0])] = parts[1]

    model = FoodResNet(num_classes=80)
    model.load_state_dict(
        torch.load(os.path.join(args.results_dir, args.experiment, "best_model.pth"),
                    map_location=device))
    model.eval().to(device)

    target_layer = model.layer4[-1].conv2
    cam = GradCAM(model=model, target_layers=[target_layer])
    _, val_transform = get_transforms("none", img_size=128)

    labels_df = pd.read_csv(os.path.join(args.data_dir, "train_labels.csv"))
    interesting = [7, 31, 39, 60, 40, 51, 70, 47, 58, 1]

    fig, axes = plt.subplots(len(interesting), 3, figsize=(12, 4 * len(interesting)))
    for i, class_id in enumerate(interesting):
        row = labels_df[labels_df["label"] == class_id].iloc[0]
        img_path = os.path.join(args.data_dir, "train_set", row["img_name"])

        raw_img = Image.open(img_path).convert("RGB").resize((128, 128))
        raw_np = np.array(raw_img) / 255.0
        input_tensor = val_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

        grayscale_cam = cam(input_tensor=input_tensor)[0, :]
        cam_image = show_cam_on_image(raw_np.astype(np.float32), grayscale_cam, use_rgb=True)

        with torch.no_grad():
            pred_class = model(input_tensor).argmax(1).item()

        true_name = class_names.get(class_id, str(class_id))
        pred_name = class_names.get(pred_class + 1, str(pred_class + 1))

        axes[i, 0].imshow(raw_np); axes[i, 0].set_title(f"True: {true_name}", fontsize=9); axes[i, 0].axis("off")
        axes[i, 1].imshow(cam_image); axes[i, 1].set_title(f"Pred: {pred_name}", fontsize=9); axes[i, 1].axis("off")
        axes[i, 2].imshow(grayscale_cam, cmap="jet"); axes[i, 2].set_title("Attention", fontsize=9); axes[i, 2].axis("off")

    plt.suptitle(f"Grad-CAM — {args.experiment}", fontsize=14)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(os.path.join("figures", f"gradcam_{args.experiment}.png"), dpi=150)
    plt.close()
    print(f"Saved gradcam_{args.experiment}.png")


if __name__ == "__main__":
    main()