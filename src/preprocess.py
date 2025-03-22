import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms

# Base and strong augmentation definitions
base_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

strong_aug_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

# Class-specific transform mapping
transform_dict = {
    0: base_transforms,        # regular
    1: strong_aug_transforms,  # semi-nude
    2: strong_aug_transforms,  # full-nude
}

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=transform_dict):
        self.df = pd.read_csv(csv_file)

        # Map string labels to integers
        label_mapping = {
            'regular': 0,
            'semi-nudity': 1,
            'full-nudity': 2
        }
        self.df['label'] = self.df['label'].map(label_mapping)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image"]
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        label = self.df.iloc[idx]["label"]

        if self.transform:
            image = self.transform[label](image)  # Class-specific transform

        return image, label


def create_loaders(batch_size=32):
    base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    train_csv = os.path.join(base_path, "train_labels.csv")
    val_csv = os.path.join(base_path, "val_labels1.csv")
    test_csv = os.path.join(base_path, "test_labels.csv")

    train_root = os.path.join(base_path, "train")
    val_root = os.path.join(base_path, "validate")
    test_root = os.path.join(base_path, "test")

    train_dataset = CustomDataset(csv_file=train_csv, root_dir=train_root, transform=transform_dict)
    val_dataset = CustomDataset(csv_file=val_csv, root_dir=val_root, transform=transform_dict)
    test_dataset = CustomDataset(csv_file=test_csv, root_dir=test_root, transform=transform_dict)

    # Compute class weights for WeightedRandomSampler
    labels = train_dataset.df["label"].values  # Already int from CustomDataset
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return {"train": train_loader, "val": val_loader, "test": test_loader}

