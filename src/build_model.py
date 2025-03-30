import torch
import torch.nn as nn
import torch.nn.functional as F
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


class SoftFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, soft_targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        focal_weight = (1 - probs) ** self.gamma
        loss = -soft_targets * focal_weight * log_probs

        if self.alpha is not None:
            loss = self.alpha.view(1, -1) * loss

        return loss.sum(dim=1).mean()


class HybridResNetSwin(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # Pretrained ResNet-50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Pretrained Swin Transformer
        self.swin = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1)
        self.swin.head = nn.Identity()

        resnet_out_dim = 2048
        swin_out_dim = 768

        self.classifier = nn.Sequential(
            nn.Linear(resnet_out_dim + swin_out_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        resnet_features = self.resnet(x).view(x.shape[0], -1)
        swin_features = self.swin(x)
        combined_features = torch.cat([resnet_features, swin_features], dim=1)
        return self.classifier(combined_features)


def compute_alpha_from_csv(csv_path):
    full_path = os.path.join(base_path, "data", csv_path)
    df = pd.read_csv(full_path)
    class_map = {'regular': 0, 'semi-nudity': 1, 'full-nudity': 2}
    labels = df['label'].map(class_map).values
    counts = Counter(labels)
    total = sum(counts.values())

    alpha = [counts.get(i, 1) / total for i in range(3)]
    alpha = [1.0 / a for a in alpha]
    alpha = torch.tensor(alpha, dtype=torch.float32)

    alpha[0] *= 1.2
    alpha = alpha / alpha.sum()

    print("Dynamic Alpha weights:", alpha.cpu().numpy())
    return alpha


def get_model(num_classes=3):
    model = HybridResNetSwin(num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    alpha = compute_alpha_from_csv("train_labels.csv").to(device)

    criterion_hard = FocalLoss(alpha=alpha).to(device)
    criterion_soft = SoftFocalLoss(alpha=alpha).to(device)

    return model.to(device), criterion_hard, criterion_soft, device
