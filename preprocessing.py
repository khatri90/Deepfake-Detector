"""
Data Preprocessing and DataLoader Setup for Real vs AI Face Detection
"""
import os
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from config import (
    REAL_IMAGES_DIR, AI_IMAGES_DIR,
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED,
    IMAGENET_MEAN, IMAGENET_STD, AUGMENTATION, DEVICE
)

# Only use pin_memory if GPU is available
PIN_MEMORY = True if DEVICE == "cuda" else False
from dataset import FaceDataset, load_image_paths_and_labels, get_class_distribution


def get_train_transforms() -> transforms.Compose:
    """
    Get data augmentation transforms for training.

    Returns:
        Composed transforms for training data
    """
    return transforms.Compose([
        # Resize to standard size
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        # Data Augmentation
        transforms.RandomHorizontalFlip(p=AUGMENTATION["horizontal_flip_prob"]),
        transforms.RandomRotation(degrees=AUGMENTATION["rotation_degrees"]),
        transforms.ColorJitter(
            brightness=AUGMENTATION["color_jitter"]["brightness"],
            contrast=AUGMENTATION["color_jitter"]["contrast"],
            saturation=AUGMENTATION["color_jitter"]["saturation"],
            hue=AUGMENTATION["color_jitter"]["hue"]
        ),

        # Convert to tensor
        transforms.ToTensor(),

        # Normalize with ImageNet statistics
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # Random erasing for regularization
        transforms.RandomErasing(p=AUGMENTATION["random_erasing_prob"])
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Get transforms for validation/test data (no augmentation).

    Returns:
        Composed transforms for validation/test data
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def split_data(
    image_paths: list,
    labels: list,
    train_size: float = TRAIN_SPLIT,
    val_size: float = VAL_SPLIT,
    test_size: float = TEST_SPLIT,
    random_state: int = RANDOM_SEED
) -> Dict[str, Tuple[list, list]]:
    """
    Split data into train, validation, and test sets with stratification.

    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        train_size: Proportion for training (default 0.7)
        val_size: Proportion for validation (default 0.15)
        test_size: Proportion for testing (default 0.15)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing split data
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Split proportions must sum to 1.0"

    # First split: separate test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels  # Maintain class balance
    )

    # Second split: separate train and validation
    val_proportion = val_size / (train_size + val_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_proportion,
        random_state=random_state,
        stratify=train_val_labels
    )

    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels)
    }


def create_datasets(data_splits: Dict) -> Dict[str, FaceDataset]:
    """
    Create PyTorch Dataset objects for each split.

    Args:
        data_splits: Dictionary from split_data()

    Returns:
        Dictionary of FaceDataset objects
    """
    datasets = {
        "train": FaceDataset(
            image_paths=data_splits["train"][0],
            labels=data_splits["train"][1],
            transform=get_train_transforms()
        ),
        "val": FaceDataset(
            image_paths=data_splits["val"][0],
            labels=data_splits["val"][1],
            transform=get_val_transforms()
        ),
        "test": FaceDataset(
            image_paths=data_splits["test"][0],
            labels=data_splits["test"][1],
            transform=get_val_transforms()
        )
    }

    return datasets


def create_dataloaders(
    datasets: Dict[str, FaceDataset],
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS
) -> Dict[str, DataLoader]:
    """
    Create DataLoader objects for each dataset split.

    Args:
        datasets: Dictionary of FaceDataset objects
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading

    Returns:
        Dictionary of DataLoader objects
    """
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,  # Only True if GPU available
            drop_last=True   # Drop incomplete batches
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY
        )
    }

    return dataloaders


def prepare_data() -> Tuple[Dict[str, FaceDataset], Dict[str, DataLoader]]:
    """
    Complete data preparation pipeline.

    Returns:
        Tuple of (datasets dict, dataloaders dict)
    """
    print("=" * 50)
    print("STARTING DATA PREPARATION")
    print("=" * 50)

    # Step 1: Load all image paths and labels
    print("\n[Step 1/4] Loading image paths...")
    image_paths, labels = load_image_paths_and_labels(REAL_IMAGES_DIR, AI_IMAGES_DIR)

    # Step 2: Show class distribution
    print("\n[Step 2/4] Analyzing class distribution...")
    dist = get_class_distribution(labels)
    for class_name, stats in dist.items():
        print(f"  {class_name}: {stats['count']} ({stats['percentage']}%)")

    # Step 3: Split data
    print("\n[Step 3/4] Splitting data into train/val/test...")
    data_splits = split_data(image_paths, labels)

    print(f"  Train: {len(data_splits['train'][0])} samples")
    print(f"  Val:   {len(data_splits['val'][0])} samples")
    print(f"  Test:  {len(data_splits['test'][0])} samples")

    # Step 4: Create datasets and dataloaders
    print("\n[Step 4/4] Creating datasets and dataloaders...")
    datasets = create_datasets(data_splits)
    dataloaders = create_dataloaders(datasets)

    print(f"\nDataLoader batch size: {BATCH_SIZE}")
    print(f"Train batches: {len(dataloaders['train'])}")
    print(f"Val batches:   {len(dataloaders['val'])}")
    print(f"Test batches:  {len(dataloaders['test'])}")

    print("\n" + "=" * 50)
    print("DATA PREPARATION COMPLETE")
    print("=" * 50)

    return datasets, dataloaders


def visualize_batch(dataloader: DataLoader, num_images: int = 8):
    """
    Visualize a batch of images from a dataloader.

    Args:
        dataloader: DataLoader to sample from
        num_images: Number of images to display
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get a batch
    images, labels = next(iter(dataloader))

    # Denormalize for visualization
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    fig, axes = plt.subplots(2, num_images // 2, figsize=(15, 6))
    axes = axes.flatten()

    for idx in range(min(num_images, len(images))):
        img = images[idx].clone()
        img = img * std + mean  # Denormalize
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        label = "Real" if labels[idx] == 0 else "AI"
        axes[idx].imshow(img)
        axes[idx].set_title(f"Label: {label}")
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig("sample_batch.png", dpi=150)
    plt.show()
    print("Sample batch saved to 'sample_batch.png'")


if __name__ == "__main__":
    # Run the complete data preparation pipeline
    datasets, dataloaders = prepare_data()

    # Test: Get a single batch
    print("\nTesting dataloader...")
    images, labels = next(iter(dataloaders["train"]))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Label values in batch: {labels.unique().tolist()}")

    # Optional: Visualize a batch (requires matplotlib)
    try:
        visualize_batch(dataloaders["train"])
    except ImportError:
        print("\nMatplotlib not installed. Skipping visualization.")
