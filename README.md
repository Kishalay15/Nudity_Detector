# Nudity Detection using Hybrid ResNet-50 + Swin Transformer

This project implements a nudity detection model using a hybrid approach that combines a Convolutional Neural Network (ResNet-50) with a Transformer-based model (Swin Transformer). The model is fine-tuned to classify images into three categories: `regular`, `semi-nudity`, and `full-nudity`.

## Features

- Uses a hybrid architecture combining **ResNet-50** and **Swin Transformer**.
- Implements **Focal Loss** and **Soft Focal Loss** to handle class imbalance.
- Data augmentation tailored for each class.
- **Mixup augmentation** for improved generalization.
- **Cosine Annealing LR scheduler** for adaptive learning rate decay.
- Early stopping mechanism based on **F1 score**.
- Generates a **confusion matrix** for evaluation.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Kishalay15/Nudity_Detector.git
   cd Nudity_Detector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## File Structure

```
├── src
│   ├── build_model.py    # Defines the hybrid model and loss functions
│   ├── preprocess.py     # Handles data loading and transformations
│   ├── train.py          # Training script
│
├── data                 # Folder containing dataset (not included in repo)
│   ├── train            # Training images
│   ├── validate         # Validation images
│   ├── test             # Test images
│   ├── train_labels.csv # Labels for training data
│   ├── val_labels1.csv  # Labels for validation data
│   ├── test_labels.csv  # Labels for test data
│
├── checkpoints          # Stores model checkpoints
├── requirements.txt     # Required dependencies
├── README.md            # Documentation
```

## Training the Model

To train the model, run:

```bash
python src/train.py
```

## Model Evaluation

After training, the best model is saved in `checkpoints/best_model.pth`. The final evaluation results include:

- Accuracy and F1 Score
- Classification Report
- Confusion Matrix (visualized)
