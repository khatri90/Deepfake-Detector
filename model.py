"""
Model Architecture for DeepFake Face Detector

Supports multiple backbones with automatic fallback:
1. EfficientNet-B0 (preferred - via timm)
2. ResNet50 (fallback - via torchvision)
3. Custom CNN (last resort fallback)
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from config import MODEL_NAME, NUM_CLASSES, PRETRAINED, FREEZE_BACKBONE, DEVICE


# ============== TRY IMPORTING BACKENDS ==============
TIMM_AVAILABLE = False
TORCHVISION_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
    print("[Model] timm library available")
except ImportError:
    print("[Model] timm not available, will use torchvision fallback")

try:
    from torchvision import models
    TORCHVISION_AVAILABLE = True
    print("[Model] torchvision available")
except ImportError:
    print("[Model] torchvision not available")


# ============== EFFICIENTNET MODEL (PRIMARY) ==============
class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 based classifier for Real vs AI face detection.

    Uses timm library for EfficientNet implementation.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = PRETRAINED,
        freeze_backbone: bool = FREEZE_BACKBONE,
        dropout_rate: float = 0.2
    ):
        super().__init__()

        self.model_name = "efficientnet_b0"

        # Load pretrained EfficientNet-B0
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0  # Remove classifier head
        )

        # Get feature dimensions
        self.num_features = self.backbone.num_features  # 1280 for B0

        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

        self._init_classifier_weights()

    def _freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"[Model] Backbone frozen")

    def unfreeze_backbone(self, unfreeze_from: Optional[str] = None):
        """
        Unfreeze backbone for fine-tuning.

        Args:
            unfreeze_from: If specified, only unfreeze layers from this point
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"[Model] Backbone unfrozen for fine-tuning")

    def _init_classifier_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        output = self.classifier(features)
        return output


# ============== RESNET50 MODEL (FALLBACK 1) ==============
class ResNet50Classifier(nn.Module):
    """
    ResNet50 based classifier - fallback when timm is not available.

    Uses torchvision implementation.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = PRETRAINED,
        freeze_backbone: bool = FREEZE_BACKBONE,
        dropout_rate: float = 0.2
    ):
        super().__init__()

        self.model_name = "resnet50"

        # Load pretrained ResNet50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        self.backbone = models.resnet50(weights=weights)

        # Get feature dimensions and remove original classifier
        self.num_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Identity()

        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

        self._init_classifier_weights()

    def _freeze_backbone(self):
        """Freeze all backbone parameters except final layer"""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        print(f"[Model] ResNet50 backbone frozen")

    def unfreeze_backbone(self, unfreeze_from: str = "layer4"):
        """
        Unfreeze backbone layers for fine-tuning.

        Args:
            unfreeze_from: Unfreeze from this layer onwards ('layer1', 'layer2', 'layer3', 'layer4')
        """
        layers = ['layer1', 'layer2', 'layer3', 'layer4']
        unfreeze_idx = layers.index(unfreeze_from) if unfreeze_from in layers else 0

        for name, param in self.backbone.named_parameters():
            for layer in layers[unfreeze_idx:]:
                if layer in name:
                    param.requires_grad = True
                    break

        print(f"[Model] ResNet50 unfrozen from {unfreeze_from}")

    def _init_classifier_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        output = self.classifier(features)
        return output


# ============== CUSTOM CNN (FALLBACK 2 - LAST RESORT) ==============
class CustomCNN(nn.Module):
    """
    Custom CNN classifier - last resort when no pretrained models available.

    Simple but effective architecture for face classification.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = 0.3
    ):
        super().__init__()

        self.model_name = "custom_cnn"

        # Convolutional layers
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 5: 14 -> 7
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize all weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def unfreeze_backbone(self, *args, **kwargs):
        """No-op for custom CNN (nothing to unfreeze)"""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# ============== MODEL FACTORY ==============
def create_model(
    model_name: str = MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = PRETRAINED,
    freeze_backbone: bool = FREEZE_BACKBONE
) -> nn.Module:
    """
    Factory function to create the appropriate model with fallbacks.

    Priority:
    1. EfficientNet-B0 (if timm available and requested)
    2. ResNet50 (if torchvision available)
    3. Custom CNN (always available)

    Args:
        model_name: Requested model ('efficientnet_b0', 'resnet50', 'custom_cnn')
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        freeze_backbone: Freeze backbone initially

    Returns:
        PyTorch model
    """
    model = None

    # Try EfficientNet first
    if model_name == "efficientnet_b0" and TIMM_AVAILABLE:
        try:
            model = EfficientNetClassifier(
                num_classes=num_classes,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone
            )
            print(f"[Model] Created EfficientNet-B0 classifier")
        except Exception as e:
            print(f"[Model] EfficientNet failed: {e}")
            model = None

    # Fallback to ResNet50
    if model is None and TORCHVISION_AVAILABLE:
        if model_name in ["efficientnet_b0", "resnet50"]:
            try:
                model = ResNet50Classifier(
                    num_classes=num_classes,
                    pretrained=pretrained,
                    freeze_backbone=freeze_backbone
                )
                print(f"[Model] Created ResNet50 classifier (fallback)")
            except Exception as e:
                print(f"[Model] ResNet50 failed: {e}")
                model = None

    # Last resort: Custom CNN
    if model is None:
        model = CustomCNN(num_classes=num_classes)
        print(f"[Model] Created Custom CNN (last resort fallback)")
        print(f"[Warning] Using custom CNN - consider installing timm or torchvision for better results")

    return model


def load_model_for_inference(
    checkpoint_path: str,
    device: str = DEVICE
) -> Tuple[nn.Module, dict]:
    """
    Load a trained model for inference.

    Args:
        checkpoint_path: Path to saved checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, checkpoint_metadata)
    """
    print(f"[Model] Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Determine model type from checkpoint
    model_name = checkpoint.get('model_name', MODEL_NAME)

    # Create model architecture
    model = create_model(
        model_name=model_name,
        pretrained=False,  # Don't need pretrained weights, loading from checkpoint
        freeze_backbone=False
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"[Model] Loaded {model_name} from epoch {checkpoint.get('epoch', 'unknown')}")

    return model, checkpoint


# ============== TEST ==============
if __name__ == "__main__":
    from utils import print_model_summary, get_device

    device = get_device()

    # Test model creation
    print("\n" + "=" * 50)
    print("Testing Model Creation")
    print("=" * 50)

    model = create_model()
    model.to(device)

    # Print summary
    print_model_summary(model)

    # Test forward pass
    print("Testing forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {torch.softmax(output[0], dim=0).tolist()}")

    print("\nModel test passed!")
