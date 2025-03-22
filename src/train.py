import torch
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from tqdm import tqdm
import sys
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


class Trainer:
    def __init__(self):
        from src.preprocess import create_loaders
        from src.build_model import get_model

        self.loaders = create_loaders()
        self.model, self.criterion, self.device = get_model()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=3e-4, weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)

        self.best_f1 = 0
        self.patience = 5
        os.makedirs("checkpoints", exist_ok=True)

    # One-hot encoding needed
    def _mixup(self, inputs, labels, alpha=0.2):
        """Mixup with One-Hot Soft Labels"""
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        index = torch.randperm(inputs.size(0)).to(self.device)

        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]

        # One-hot encode labels → soft label mix
        labels_onehot = F.one_hot(labels, num_classes=3).float()
        mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[index]
        return mixed_inputs, mixed_labels  # Return soft labels


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
                    # Compute soft cross-entropy manually
                    loss = -(soft_labels * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()
                else:
                    loss = self.criterion(outputs, labels)  # Use FocalLoss here

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

            # Training Phase
            train_metrics = self._run_epoch(self.loaders["train"])
            val_metrics = self._run_epoch(self.loaders["val"], training=False)
            self.scheduler.step()

            print(f"Train Accuracy: {train_metrics['accuracy'] * 100:.2f}% | F1 Score: {train_metrics['f1']:.2f}")
            print(f"Validation Accuracy: {val_metrics['accuracy'] * 100:.2f}% | F1 Score: {val_metrics['f1']:.2f}")
            print("Validation Confusion Matrix:")
            print(val_metrics["cm"])

            # Save best model based on validation F1 score
            if val_metrics["f1"] > self.best_f1:
                self.best_f1 = val_metrics["f1"]
                torch.save(self.model.state_dict(), f"checkpoints/best_model_epoch{epoch}.pth")
                self.patience = 5
            else:
                self.patience -= 1

            if self.patience <= 0:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load the best model
        import glob
        best_model_files = sorted(glob.glob("checkpoints/best_model_epoch*.pth"))
        if best_model_files:
            best_model_path = best_model_files[-1]
            print(f"\nLoading best model from {best_model_path}")
            self.model.load_state_dict(torch.load(best_model_path))
        else:
            print("No model checkpoint found!")

        # Final Test Phase
        test_metrics = self._run_epoch(self.loaders["test"], training=False)
        print("\nFinal Test Results:")
        print(f"Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
        print(f"Test Macro F1 Score: {test_metrics['f1']:.2f}")
        print("Test Confusion Matrix:")
        print(test_metrics["cm"])



if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(epochs=30)
