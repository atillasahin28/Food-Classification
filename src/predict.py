"""
Generate Kaggle submission CSV.
Usage: python src/predict.py --experiment combined
"""

import os
import sys
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from model import FoodResNet
from dataset import FoodTestDataset, get_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="combined")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FoodResNet(num_classes=80)
    model_path = os.path.join(args.results_dir, args.experiment, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    _, val_transform = get_transforms("none", args.img_size)
    test_dataset = FoodTestDataset(
        img_dir=os.path.join(args.data_dir, "test_set"),
        transform=val_transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    all_names = []
    all_preds = []

    with torch.no_grad():
        for images, img_names in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            preds = preds + 1  # Convert 0-79 back to 1-80
            all_names.extend(img_names)
            all_preds.extend(preds.tolist())

    submission = pd.DataFrame({"img_name": all_names, "label": all_preds})
    os.makedirs("submissions", exist_ok=True)
    out_path = os.path.join("submissions", f"submission_{args.experiment}.csv")
    submission.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(submission)} predictions)")


if __name__ == "__main__":
    main()