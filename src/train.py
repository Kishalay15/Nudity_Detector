import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from tqdm import tqdm
import sys
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import create_loaders
from src.build_model import get_model


class Trainer:
    def __init__(self):
        self.loaders = create_loaders()
        self.model, self.criterion_hard, self.criterion_soft, self.device = get_model()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=3e-4, weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)

        self.best_f1 = 0
        self.patience = 5
        os.makedirs("checkpoints", exist_ok=True)

    def _mixup(self, inputs, labels, alpha=0.2):
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        index = torch.randperm(inputs.size(0)).to(self.device)

        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        labels_onehot = F.one_hot(labels, num_classes=3).float()
        mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[index]
        return mixed_inputs, mixed_labels

    def _run_epoch(self, loader, training=True):
        self.model.train(training)
        total_loss = 0
        all_preds, all_labels = [], []
        mixup_prob = 0.5

        with torch.set_grad_enabled(training):
            for inputs, labels in tqdm(loader, desc="Training" if training else "Validating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device, dtype=torch.long)

                mixup_applied = False
                if training and torch.rand(1).item() < mixup_prob:
                    inputs, soft_labels = self._mixup(inputs, labels)
                    mixup_applied = True

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                if mixup_applied:
                    loss = self.criterion_soft(outputs, soft_labels)
                else:
                    loss = self.criterion_hard(outputs, labels)

                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = sum(1 for x, y in zip(all_preds, all_labels) if x == y) / len(all_labels)

        return {
            "loss": total_loss / len(loader),
            "accuracy": accuracy,
            "report": classification_report(
                all_labels, all_preds, target_names=["regular", "semi-nude", "full-nude"], output_dict=True, zero_division=0
            ),
            "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
            "cm": confusion_matrix(all_labels, all_preds),
        }

    def train(self, epochs=30):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            train_metrics = self._run_epoch(self.loaders["train"])
            val_metrics = self._run_epoch(self.loaders["val"], training=False)
            self.scheduler.step()

            print(f"Train Acc: {train_metrics['accuracy']*100:.2f}% | F1: {train_metrics['f1']:.3f}")
            print(f"Val   Acc: {val_metrics['accuracy']*100:.2f}% | F1: {val_metrics['f1']:.3f}")

            if val_metrics["f1"] > self.best_f1:
                self.best_f1 = val_metrics["f1"]
                torch.save(self.model.state_dict(), "checkpoints/best_model.pth")
                print(f"✅ Best model saved at epoch {epoch+1}")
                self.patience = 5
            else:
                self.patience -= 1
                print(f"Patience left: {self.patience}")

            if self.patience <= 0:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break

        best_model_path = "checkpoints/best_model.pth"
        if os.path.exists(best_model_path):
            print(f"\nLoading best model from {best_model_path}")
            self.model.load_state_dict(torch.load(best_model_path))
        else:
            print("No model checkpoint found!")

        test_metrics = self._run_epoch(self.loaders["test"], training=False)
        print("\n📄 Final Test Results:")
        print(f"Test Acc: {test_metrics['accuracy']*100:.2f}%")
        print(f"Test F1 : {test_metrics['f1']:.3f}")
        print("Confusion Matrix:")
        print(test_metrics["cm"])

        self._plot_test_metrics(test_metrics)

    def _plot_test_metrics(self, test_metrics):
        plt.figure(figsize=(6, 4))
        classes = ["regular", "semi-nude", "full-nude"]
        cm = test_metrics["cm"]

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Test Confusion Matrix")
        plt.colorbar()
        tick_marks = range(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2. else "black")

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(epochs=30)
