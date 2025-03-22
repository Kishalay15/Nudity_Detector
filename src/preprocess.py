import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision import transforms
import torch

# Define dataset directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ✅ Define Transformations Separately
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_transform(is_train):
    return train_transforms if is_train else val_transforms

class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, is_train=True):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.class_map = {'regular': 0, 'semi-nudity': 1, 'full-nudity': 2}
        self.samples = self._filter_valid_images()
        self.class_weights = self._compute_class_weights()
        self.is_train = is_train

    def _compute_class_weights(self):
        class_counts = np.bincount(
            [self.class_map[self.df.iloc[idx, 1]] for idx in self.samples], minlength=3
        )
        class_counts = np.maximum(class_counts, 1)  # Avoid division by zero
        return 1.0 / class_counts


    def _filter_valid_images(self):
        valid_samples = []
        for idx in range(len(self.df)):
            img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        img.convert("RGB")
                    valid_samples.append(idx)
                except Exception:
                    continue  # ✅ Skip corrupt images silently (no print)
        return valid_samples


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idx = self.samples[idx]
        img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        label_str = self.df.iloc[idx, 1]
        label = self.class_map[label_str]
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                transform = get_transform(self.is_train)

                img = transform(img)

                # Extra augmentation for minority class 'regular'
                if self.is_train and label_str == 'regular':
                    extra_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4)
                    img = extra_jitter(img)

            return img, label
        except Exception as e:
            print(f"Error loading image: {img_path} - {e}")
            return None, None

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)  # ✅ Convert to long

def create_loaders():
    datasets = {
        "train": ImageDataset(f"{DATA_DIR}/train_labels.csv", f"{DATA_DIR}/train", is_train=True),
        "val": ImageDataset(f"{DATA_DIR}/val_labels1.csv", f"{DATA_DIR}/validate", is_train=False),
        "test": ImageDataset(f"{DATA_DIR}/test_labels.csv", f"{DATA_DIR}/test", is_train=False),
    }

    class_weights = datasets["train"].class_weights
    weights = [class_weights[datasets["train"].class_map[datasets["train"].df.iloc[idx, 1]]]
               for idx in datasets["train"].samples]

    # Amplify weight for 'regular' class
    amplified_weights = []
    for i, idx in enumerate(datasets["train"].samples):
        label = datasets["train"].df.iloc[idx, 1]
        weight = weights[i]
        if label == 'regular':
            weight *= 4  # ⬅️ Amplify oversampling for 'regular'
        amplified_weights.append(weight)

    sampler = WeightedRandomSampler(amplified_weights, len(amplified_weights), replacement=True)


    return {
    "train": DataLoader(datasets["train"], batch_size=32, sampler=sampler, collate_fn=collate_fn, num_workers=4, drop_last=True),
    "val": DataLoader(datasets["val"], batch_size=32, collate_fn=collate_fn, num_workers=4, drop_last=False),  # ✅ Allow smaller batch
    "test": DataLoader(datasets["test"], batch_size=32, collate_fn=collate_fn, num_workers=4, drop_last=False),  # ✅ Allow smaller batch
    }

