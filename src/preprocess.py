import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, df, img_dir, transform_dict):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform_dict = transform_dict
        self.label_mapping = {'regular': 0, 'semi-nudity': 1, 'full-nudity': 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['image']
        label_str = row['label']

        img_path = os.path.join(self.img_dir, image_name)
        img = Image.open(img_path).convert('RGB')

        transform = self.transform_dict[label_str]
        img = transform(img)

        label = self.label_mapping[label_str]
        return img, label

def create_loaders(batch_size=32, num_workers=2):
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    # CSV Paths
    train_csv = os.path.join(data_dir, "train_labels.csv")
    val_csv = os.path.join(data_dir, "val_labels1.csv")
    test_csv = os.path.join(data_dir, "test_labels.csv")

    # Image dirs
    train_img_dir = os.path.join(data_dir, "train")
    val_img_dir = os.path.join(data_dir, "validate")
    test_img_dir = os.path.join(data_dir, "test")

    # Read CSVs
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Class-based Augmentations
    transform_dict = {
        'regular': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
        ]),
        'semi-nudity': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'full-nudity': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
    }

    # Datasets
    train_dataset = CustomDataset(train_df, train_img_dir, transform_dict)
    val_dataset = CustomDataset(val_df, val_img_dir, transform_dict)
    test_dataset = CustomDataset(test_df, test_img_dir, transform_dict)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {"train": train_loader, "val": val_loader, "test": test_loader}
