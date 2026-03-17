"""
PyTorch Dataset for the food classification task.
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class FoodDataset(Dataset):
    """
    Loads food images and labels.
    Labels are shifted from 1-80 to 0-79 for PyTorch cross-entropy.
    """

    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # IMPORTANT: Kaggle labels are 1-80, PyTorch expects 0-79
        self.df["label"] = self.df["label"] - 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["img_name"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["label"])
        return image, label


class FoodTestDataset(Dataset):
    """
    Loads test images (no labels) for generating predictions.
    """

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name


def get_transforms(augmentation="none", img_size=128):
    """
    Returns train and validation transforms.

    augmentation options:
        "none"         - no augmentation (baseline)
        "geometric"    - flips + rotation + resized crop
        "color_jitter" - color changes
        "cutout"       - random erasing
        "combined"     - best combination
    """

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    # Validation transform is always the same
    val_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        normalize,
    ])

    # Build training transform based on augmentation type
    train_ops = [T.Resize((img_size, img_size))]

    if augmentation == "none":
        pass

    elif augmentation == "geometric":
        train_ops += [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(15),
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        ]

    elif augmentation == "color_jitter":
        train_ops += [
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ]

    elif augmentation == "cutout":
        train_ops += [
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        ]
        train_transform = T.Compose(train_ops)
        return train_transform, val_transform

    elif augmentation == "combined":
        train_ops += [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(15),
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ]
        train_transform = T.Compose(train_ops)
        return train_transform, val_transform

    train_ops += [T.ToTensor(), normalize]
    train_transform = T.Compose(train_ops)

    return train_transform, val_transform