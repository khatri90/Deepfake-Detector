"""
Configuration file for Real vs AI Face Detection Project
"""
import os

# ============== PATHS ==============
BASE_DIR = r"C:\Users\PMLS\Downloads\Real Or AI image detection\Human Faces"
REAL_IMAGES_DIR = os.path.join(BASE_DIR, "Real Images")
AI_IMAGES_DIR = os.path.join(BASE_DIR, "AI-Generated Images")

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# ============== IMAGE SETTINGS ==============
IMAGE_SIZE = 224  # Standard input size for pre-trained models
IMAGE_CHANNELS = 3

# ============== DATASET SETTINGS ==============
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ============== CLASS LABELS ==============
CLASSES = {
    0: "Real",
    1: "AI-Generated"
}
NUM_CLASSES = 2

# ============== TRAINING HYPERPARAMETERS ==============
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

# ============== MODEL SETTINGS ==============
MODEL_NAME = "efficientnet_b0"  # Options: "efficientnet_b0", "resnet50", "xception"
PRETRAINED = True
FREEZE_BACKBONE = True  # Freeze early layers initially
UNFREEZE_AT_EPOCH = 5   # Unfreeze backbone after this epoch for fine-tuning

# ============== DATA AUGMENTATION SETTINGS ==============
AUGMENTATION = {
    "horizontal_flip_prob": 0.5,
    "rotation_degrees": 15,
    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1
    },
    "random_erasing_prob": 0.1
}

# ============== NORMALIZATION (ImageNet stats) ==============
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============== DEVICE ==============
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_directories():
    """Create necessary output directories"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    print(f"Directories created at: {OUTPUT_DIR}")


if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Real Images Directory: {REAL_IMAGES_DIR}")
    print(f"AI Images Directory: {AI_IMAGES_DIR}")
    print(f"Device: {DEVICE}")
    create_directories()
