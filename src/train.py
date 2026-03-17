"""
Main training script.

Usage:
    python src/train.py --augmentation none --name baseline
    python src/train.py --augmentation geometric --name geometric
    python src/train.py --augmentation color_jitter --name color_jitter
    python src/train.py --augmentation cutout --name cutout
    python src/train.py --augmentation mixup --name mixup
    python src/train.py --augmentation cutmix --name cutmix
    python src/train.py --augmentation combined --name combined
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# Add src to path so imports work from any directory
sys.path.insert(0, os.path.dirname(__file__))

from dataset import FoodDataset, get_transforms
from model import FoodResNet
from mixup import mixup_data, mixup_criterion, cutmix_data, cutmix_criterion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--augmentation", type=str, default="none",
                        choices=["none", "geometric", "color_jitter", "cutout",
                                 "mixup", "cutmix", "combined"])
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--results_dir", type=str, default="results")
    return parser.parse_args()


class TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset with a specific transform."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def train_one_epoch(model, loader, criterion, optimizer, device, augmentation="none"):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        if augmentation == "mixup":
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(labels_a).sum().item()
                        + (1 - lam) * predicted.eq(labels_b).sum().item())
        elif augmentation == "cutmix":
            images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=1.0)
            outputs = model(images)
            loss = cutmix_criterion(criterion, outputs, labels_a, labels_b, lam)
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(labels_a).sum().item()
                        + (1 - lam) * predicted.eq(labels_b).sum().item())
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=80):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        _, predicted = outputs.max(1)
        running_loss += loss.item() * images.size(0)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i].item() == label:
                class_correct[label] += 1

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    per_class_acc = np.where(class_total > 0, class_correct / class_total, 0.0)
    return (running_loss / total, correct / total, per_class_acc,
            np.array(all_preds), np.array(all_labels))


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = os.path.join(args.results_dir, args.name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # For MixUp/CutMix, transforms are "none" (applied in training loop)
    aug_for_transforms = args.augmentation
    if args.augmentation in ["mixup", "cutmix"]:
        aug_for_transforms = "none"

    train_transform, val_transform = get_transforms(aug_for_transforms, args.img_size)

    full_dataset = FoodDataset(
        csv_file=os.path.join(args.data_dir, "train_labels.csv"),
        img_dir=os.path.join(args.data_dir, "train_set"),
        transform=None,
    )

    # Stratified 85/15 split
    labels_array = full_dataset.df["label"].values
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels_array)), labels_array))

    train_data = TransformSubset(Subset(full_dataset, train_idx), train_transform)
    val_data = TransformSubset(Subset(full_dataset, val_idx), val_transform)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    model = FoodResNet(num_classes=80, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.augmentation)
        val_loss, val_acc, per_class_acc, preds, labels = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]

        print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  ({elapsed:.1f}s)")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            np.save(os.path.join(save_dir, "best_per_class_acc.npy"), per_class_acc)
            np.save(os.path.join(save_dir, "best_preds.npy"), preds)
            np.save(os.path.join(save_dir, "best_labels.npy"), labels)
            print(f"  >>> New best: {val_acc:.4f}")

    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f)

    print(f"\nDone! Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()