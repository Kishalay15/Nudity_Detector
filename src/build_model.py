import torch
import torch.nn as nn
from torchvision import models

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

def get_model(num_classes=3):
    model = CustomResNet(num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Normalized class weights
    class_weights = torch.tensor([1.0, 3.0, 1.5], dtype=torch.float32)
    class_weights /= class_weights.sum()
    class_weights = class_weights.to(device)

    criterion = FocalLoss(alpha=class_weights).to(device)
    return model.to(device), criterion, device
