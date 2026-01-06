"""
Inference Script for DeepFake Face Detector

Features:
- Single image prediction
- Batch prediction on folder
- Confidence scores
- Optional visualization
- JSON output support
"""
import os
import sys
import argparse
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Try importing tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from config import DEVICE, MODEL_DIR, CLASSES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from model import load_model_for_inference, create_model
from preprocessing import get_val_transforms
from utils import get_device


class Predictor:
    """
    Predictor class for inference on new images.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = DEVICE,
        class_names: Dict[int, str] = CLASSES
    ):
        self.device = device
        self.class_names = class_names

        # Load model
        if model_path and os.path.exists(model_path):
            self.model, self.checkpoint = load_model_for_inference(model_path, device)
        else:
            # Try default path
            default_path = os.path.join(MODEL_DIR, 'best_model.pth')
            if os.path.exists(default_path):
                self.model, self.checkpoint = load_model_for_inference(default_path, device)
            else:
                raise FileNotFoundError(
                    f"No model checkpoint found. Please train a model first or specify --checkpoint path."
                )

        self.model.eval()

        # Get transforms
        self.transform = get_val_transforms()

        print(f"[Predictor] Model loaded successfully")
        print(f"[Predictor] Device: {device}")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed tensor ready for inference
        """
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        tensor = self.transform(image)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor

    @torch.no_grad()
    def predict_single(self, image_path: str) -> Dict:
        """
        Predict on a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with prediction results
        """
        if not os.path.exists(image_path):
            return {
                'error': f'File not found: {image_path}',
                'image_path': image_path
            }

        try:
            # Preprocess
            tensor = self.preprocess_image(image_path).to(self.device)

            # Inference
            output = self.model(tensor)
            probabilities = F.softmax(output, dim=1)[0]

            # Get prediction
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[predicted_class].item()

            # Prepare results
            result = {
                'image_path': image_path,
                'prediction': self.class_names[predicted_class],
                'predicted_class_id': predicted_class,
                'confidence': confidence,
                'probabilities': {
                    self.class_names[i]: prob.item()
                    for i, prob in enumerate(probabilities)
                },
                'is_ai_generated': predicted_class == 1
            }

            return result

        except Exception as e:
            return {
                'error': str(e),
                'image_path': image_path
            }

    def predict_batch(self, image_paths: List[str], show_progress: bool = True) -> List[Dict]:
        """
        Predict on multiple images.

        Args:
            image_paths: List of image file paths
            show_progress: Show progress bar

        Returns:
            List of prediction results
        """
        results = []

        if show_progress and TQDM_AVAILABLE:
            pbar = tqdm(image_paths, desc="Predicting")
        else:
            pbar = image_paths

        for path in pbar:
            result = self.predict_single(path)
            results.append(result)

        return results

    def predict_folder(self, folder_path: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')) -> List[Dict]:
        """
        Predict on all images in a folder.

        Args:
            folder_path: Path to folder containing images
            extensions: Valid image extensions

        Returns:
            List of prediction results
        """
        if not os.path.isdir(folder_path):
            raise ValueError(f"Not a valid directory: {folder_path}")

        # Collect image paths
        image_paths = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(extensions):
                image_paths.append(os.path.join(folder_path, filename))

        print(f"[Predictor] Found {len(image_paths)} images in {folder_path}")

        return self.predict_batch(image_paths)


def visualize_prediction(image_path: str, result: Dict, save_path: Optional[str] = None) -> None:
    """
    Visualize prediction result.

    Args:
        image_path: Path to the image
        result: Prediction result dictionary
        save_path: Optional path to save visualization
    """
    if not PLOTTING_AVAILABLE:
        print("[Visualize] matplotlib not available, skipping visualization")
        return

    # Load image
    image = Image.open(image_path)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Show image
    axes[0].imshow(image)
    axes[0].set_title(f"Prediction: {result['prediction']}\nConfidence: {result['confidence']*100:.1f}%")
    axes[0].axis('off')

    # Show probabilities
    probs = result['probabilities']
    colors = ['green' if 'Real' in k else 'red' for k in probs.keys()]
    bars = axes[1].barh(list(probs.keys()), list(probs.values()), color=colors)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel('Probability')
    axes[1].set_title('Class Probabilities')

    # Add value labels
    for bar, val in zip(bars, probs.values()):
        axes[1].text(val + 0.02, bar.get_y() + bar.get_height()/2,
                     f'{val*100:.1f}%', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Visualize] Saved to: {save_path}")

    plt.show()


def print_result(result: Dict) -> None:
    """Print prediction result in a formatted way"""
    if 'error' in result:
        print(f"\n[Error] {result['image_path']}: {result['error']}")
        return

    print(f"\n{'='*50}")
    print(f"Image: {os.path.basename(result['image_path'])}")
    print(f"{'='*50}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"\nClass Probabilities:")
    for class_name, prob in result['probabilities'].items():
        indicator = "<<<" if class_name == result['prediction'] else ""
        print(f"  {class_name}: {prob*100:.2f}% {indicator}")

    if result['is_ai_generated']:
        print(f"\n[!] This image appears to be AI-GENERATED")
    else:
        print(f"\n[✓] This image appears to be REAL")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DeepFake Face Detection Inference')

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str,
                             help='Path to a single image')
    input_group.add_argument('--folder', type=str,
                             help='Path to folder of images')

    # Model options
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')

    # Output options
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path for results')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization for predictions')
    parser.add_argument('--save-viz', type=str, default=None,
                        help='Save visualization to this path')

    # Other options
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions')

    return parser.parse_args()


def main():
    """Main inference function"""
    args = parse_args()

    # Get device
    device = get_device()

    # Create predictor
    try:
        predictor = Predictor(
            model_path=args.checkpoint,
            device=device
        )
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        print("[Error] Train a model first with: python train.py")
        sys.exit(1)

    # Run prediction
    if args.image:
        # Single image prediction
        result = predictor.predict_single(args.image)
        print_result(result)

        # Visualize if requested
        if args.visualize or args.save_viz:
            if 'error' not in result:
                visualize_prediction(args.image, result, save_path=args.save_viz)

        results = [result]

    elif args.folder:
        # Folder prediction
        results = predictor.predict_folder(args.folder)

        # Print summary
        print(f"\n{'='*60}")
        print("BATCH PREDICTION SUMMARY")
        print(f"{'='*60}")

        real_count = sum(1 for r in results if not r.get('is_ai_generated', False) and 'error' not in r)
        ai_count = sum(1 for r in results if r.get('is_ai_generated', False) and 'error' not in r)
        error_count = sum(1 for r in results if 'error' in r)

        print(f"\nTotal images: {len(results)}")
        print(f"  Real faces:     {real_count}")
        print(f"  AI-generated:   {ai_count}")
        print(f"  Errors:         {error_count}")

        # Print individual results
        print(f"\nDetailed Results:")
        for result in results:
            if 'error' not in result:
                status = "AI" if result['is_ai_generated'] else "Real"
                conf = result['confidence'] * 100
                print(f"  [{status:4}] {os.path.basename(result['image_path']):30} ({conf:.1f}%)")
            else:
                print(f"  [ERR ] {os.path.basename(result['image_path']):30} - {result['error']}")

    # Save results to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[Main] Results saved to: {args.output}")

    print("\n[Main] Inference complete!")


# ============== QUICK PREDICTION FUNCTION ==============
def quick_predict(image_path: str, model_path: Optional[str] = None) -> Dict:
    """
    Quick prediction function for programmatic use.

    Args:
        image_path: Path to image
        model_path: Optional path to model checkpoint

    Returns:
        Prediction result dictionary

    Example:
        >>> from predict import quick_predict
        >>> result = quick_predict("path/to/face.jpg")
        >>> print(f"Prediction: {result['prediction']} ({result['confidence']*100:.1f}%)")
    """
    device = get_device()
    predictor = Predictor(model_path=model_path, device=device)
    return predictor.predict_single(image_path)


if __name__ == "__main__":
    main()
