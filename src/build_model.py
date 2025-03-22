import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
from collections import Counter
import os

base_path = os.path.dirname(os.path.dirname(__file__))

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets = targets.long()
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()

# build_model.py (add this class below FocalLoss)

class SoftFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, soft_targets):
        """
        inputs: logits (batch_size, num_classes)
        soft_targets: probabilities (batch_size, num_classes)
        """
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        focal_weight = (1 - probs) ** self.gamma
        loss = -soft_targets * focal_weight * log_probs

        if self.alpha is not None:
            loss = self.alpha.unsqueeze(0) * loss  # broadcast alpha across batch

        return loss.sum(dim=1).mean()

class CustomResNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze only layer1 and layer2
        for param in self.model.layer1.parameters():
            param.requires_grad = False
        for param in self.model.layer2.parameters():
            param.requires_grad = False

        # Unfreeze layer3 and layer4 (fine-tuning)
        for param in self.model.layer3.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace the fully connected layer
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)

def compute_alpha_from_csv(csv_path):
    full_path = os.path.join(base_path, "data", csv_path)
    df = pd.read_csv(full_path)
    class_map = {'regular': 0, 'semi-nudity': 1, 'full-nudity': 2}
    labels = df['label'].map(class_map).values
    counts = Counter(labels)
    total = sum(counts.values())

    # Inverse class frequency
    alpha = [counts.get(i, 1) / total for i in range(3)]
    alpha = [1.0 / a for a in alpha]  # Inverse frequency
    alpha = torch.tensor(alpha, dtype=torch.float32)

    # Optional: Boost 'regular' class if needed
    alpha[0] *= 1.2  # Adjust boost factor as needed

    # Normalize weights to sum = 1
    alpha = alpha / alpha.sum()
    print("Dynamic Alpha weights:", alpha.cpu().numpy())
    return alpha

def get_model(num_classes=3):
    model = CustomResNet(num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    alpha = compute_alpha_from_csv("train_labels.csv").to(device)

    criterion_hard = FocalLoss(alpha=alpha).to(device)
    criterion_soft = SoftFocalLoss(alpha=alpha).to(device)

    return model.to(device), criterion_hard, device

