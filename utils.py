"""
Utility Functions for Real vs AI Face Detection Project
"""
import os
import random
import numpy as np
import torch
from datetime import datetime
from typing import Optional
import json


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to: {seed}")


def get_device() -> torch.device:
    """
    Get the best available device (GPU/CPU).

    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": round(trainable / total * 100, 2)
    }


def print_model_summary(model: torch.nn.Module, input_size: tuple = (1, 3, 224, 224)):
    """
    Print model summary including parameter counts.

    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
    """
    params = count_parameters(model)

    print("\n" + "=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(f"Total parameters:     {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,} ({params['trainable_pct']}%)")
    print(f"Frozen parameters:    {params['frozen']:,}")
    print("=" * 50 + "\n")


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss, 'max' for accuracy
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\nEarly stopping triggered! No improvement for {self.patience} epochs.")
                print(f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}")

        return self.early_stop

    def _is_improvement(self, score: float) -> bool:
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


class MetricsTracker:
    """
    Track and store training metrics.
    """

    def __init__(self):
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": []
        }

    def update(self, train_loss: float, train_acc: float,
               val_loss: float, val_acc: float, lr: float):
        """Update metrics for current epoch"""
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        self.history["learning_rate"].append(lr)

    def get_best(self) -> dict:
        """Get best metrics"""
        if not self.history["val_acc"]:
            return {}

        best_epoch = np.argmax(self.history["val_acc"])
        return {
            "best_epoch": int(best_epoch),
            "best_val_acc": self.history["val_acc"][best_epoch],
            "best_val_loss": self.history["val_loss"][best_epoch]
        }

    def save(self, filepath: str):
        """Save metrics to JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Metrics saved to: {filepath}")

    def plot(self, save_path: Optional[str] = None):
        """Plot training history"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss plot
        axes[0].plot(self.history["train_loss"], label="Train Loss")
        axes[0].plot(self.history["val_loss"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy plot
        axes[1].plot(self.history["train_acc"], label="Train Acc")
        axes[1].plot(self.history["val_acc"], label="Val Acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training & Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True)

        # Learning rate plot
        axes[2].plot(self.history["learning_rate"])
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_title("Learning Rate Schedule")
        axes[2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to: {save_path}")

        plt.show()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    filepath: str
):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None) -> dict:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into

    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(filepath, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model weights loaded from epoch {checkpoint['epoch']}")

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Optimizer state loaded")

    return checkpoint


def calculate_class_weights(labels: list) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.

    Args:
        labels: List of labels

    Returns:
        Tensor of class weights
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    # Inverse frequency weighting
    weights = total / (len(unique) * counts)
    weights = torch.tensor(weights, dtype=torch.float32)

    print(f"Class weights: {weights.tolist()}")
    return weights


def get_timestamp() -> str:
    """Get current timestamp string for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...\n")

    # Test seed setting
    set_seed(42)

    # Test device detection
    device = get_device()

    # Test early stopping
    print("\nTesting EarlyStopping...")
    early_stop = EarlyStopping(patience=3, mode="min")
    scores = [0.5, 0.4, 0.45, 0.46, 0.47, 0.48]  # Simulated val losses
    for epoch, score in enumerate(scores):
        if early_stop(score, epoch):
            print(f"Would stop at epoch {epoch}")
            break

    print("\nAll utility tests passed!")
