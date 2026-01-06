"""
Custom PyTorch Dataset for Real vs AI Face Detection
"""
import os
from typing import Tuple, List, Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class FaceDataset(Dataset):
    """
    Custom Dataset for loading Real and AI-Generated face images.

    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels (0=Real, 1=AI)
        transform: Optional transforms to apply to images
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        assert len(image_paths) == len(labels), \
            f"Mismatch: {len(image_paths)} images vs {len(labels)} labels"

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Get label
        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_sample_info(self, idx: int) -> dict:
        """Get detailed information about a sample"""
        return {
            "path": self.image_paths[idx],
            "label": self.labels[idx],
            "label_name": "Real" if self.labels[idx] == 0 else "AI-Generated"
        }


def load_image_paths_and_labels(
    real_dir: str,
    ai_dir: str,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
) -> Tuple[List[str], List[int]]:
    """
    Load all image paths and create corresponding labels.

    Args:
        real_dir: Path to directory containing real images
        ai_dir: Path to directory containing AI-generated images
        extensions: Tuple of valid image extensions

    Returns:
        Tuple of (image_paths, labels)
    """
    image_paths = []
    labels = []

    # Load real images (label = 0)
    print(f"Loading real images from: {real_dir}")
    real_count = 0
    for filename in os.listdir(real_dir):
        if filename.lower().endswith(extensions):
            image_paths.append(os.path.join(real_dir, filename))
            labels.append(0)  # Real = 0
            real_count += 1
    print(f"  Found {real_count} real images")

    # Load AI-generated images (label = 1)
    print(f"Loading AI images from: {ai_dir}")
    ai_count = 0
    for filename in os.listdir(ai_dir):
        if filename.lower().endswith(extensions):
            image_paths.append(os.path.join(ai_dir, filename))
            labels.append(1)  # AI = 1
            ai_count += 1
    print(f"  Found {ai_count} AI-generated images")

    print(f"Total images loaded: {len(image_paths)}")

    return image_paths, labels


def get_class_distribution(labels: List[int]) -> dict:
    """
    Calculate class distribution statistics.

    Args:
        labels: List of labels

    Returns:
        Dictionary with class counts and percentages
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    distribution = {}
    for label, count in zip(unique, counts):
        label_name = "Real" if label == 0 else "AI-Generated"
        distribution[label_name] = {
            "count": int(count),
            "percentage": round(count / total * 100, 2)
        }

    return distribution


def verify_images(image_paths: List[str], verbose: bool = False) -> Tuple[List[str], List[int]]:
    """
    Verify that all images can be opened and are valid.

    Args:
        image_paths: List of image paths to verify
        verbose: Print details about corrupted images

    Returns:
        Tuple of (valid_paths, corrupted_indices)
    """
    valid_paths = []
    corrupted_indices = []

    print("Verifying images...")
    for idx, path in enumerate(image_paths):
        try:
            with Image.open(path) as img:
                img.verify()
            valid_paths.append(path)
        except Exception as e:
            corrupted_indices.append(idx)
            if verbose:
                print(f"  Corrupted: {path} - {str(e)}")

    if corrupted_indices:
        print(f"Found {len(corrupted_indices)} corrupted images")
    else:
        print("All images verified successfully!")

    return valid_paths, corrupted_indices


if __name__ == "__main__":
    # Test the dataset module
    from config import REAL_IMAGES_DIR, AI_IMAGES_DIR

    # Load paths and labels
    paths, labels = load_image_paths_and_labels(REAL_IMAGES_DIR, AI_IMAGES_DIR)

    # Show distribution
    dist = get_class_distribution(labels)
    print("\nClass Distribution:")
    for class_name, stats in dist.items():
        print(f"  {class_name}: {stats['count']} ({stats['percentage']}%)")
