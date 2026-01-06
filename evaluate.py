"""
Evaluation Script for DeepFake Face Detector

Features:
- Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
- Confusion matrix visualization
- Per-class metrics
- ROC curve plotting
- Misclassified samples analysis
"""
import os
import argparse
from typing import Dict, List, Tuple, Optional
import json

import torch
import torch.nn as nn
import numpy as np

# Try importing sklearn for metrics
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score, roc_curve
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[Evaluate] sklearn not available, using basic metrics only")

# Try importing matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("[Evaluate] matplotlib/seaborn not available, skipping visualizations")

# Try importing tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from config import DEVICE, MODEL_DIR, LOGS_DIR, CLASSES, create_directories
from model import load_model_for_inference, create_model
from preprocessing import prepare_data
from utils import get_device


class Evaluator:
    """
    Evaluator class for comprehensive model evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader,
        device: str = DEVICE,
        class_names: Dict[int, str] = CLASSES
    ):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names

        # Results storage
        self.all_labels = []
        self.all_predictions = []
        self.all_probabilities = []

    @torch.no_grad()
    def run_inference(self) -> None:
        """Run inference on the entire test set"""
        print("\n[Evaluator] Running inference on test set...")

        if TQDM_AVAILABLE:
            pbar = tqdm(self.test_loader, desc="Evaluating")
        else:
            pbar = self.test_loader
            print("  Processing batches...")

        for images, labels in pbar:
            images = images.to(self.device)

            # Forward pass
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            # Store results
            self.all_labels.extend(labels.cpu().numpy())
            self.all_predictions.extend(predictions.cpu().numpy())
            self.all_probabilities.extend(probabilities.cpu().numpy())

        self.all_labels = np.array(self.all_labels)
        self.all_predictions = np.array(self.all_predictions)
        self.all_probabilities = np.array(self.all_probabilities)

        print(f"[Evaluator] Processed {len(self.all_labels)} samples")

    def calculate_basic_metrics(self) -> Dict:
        """Calculate basic accuracy metrics without sklearn"""
        correct = (self.all_predictions == self.all_labels).sum()
        total = len(self.all_labels)
        accuracy = correct / total

        # Per-class accuracy
        per_class_acc = {}
        for class_id, class_name in self.class_names.items():
            mask = self.all_labels == class_id
            if mask.sum() > 0:
                class_correct = (self.all_predictions[mask] == self.all_labels[mask]).sum()
                per_class_acc[class_name] = class_correct / mask.sum()

        return {
            'accuracy': accuracy,
            'total_samples': total,
            'correct_samples': int(correct),
            'per_class_accuracy': per_class_acc
        }

    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive metrics"""
        if not SKLEARN_AVAILABLE:
            return self.calculate_basic_metrics()

        # Basic metrics
        accuracy = accuracy_score(self.all_labels, self.all_predictions)
        precision = precision_score(self.all_labels, self.all_predictions, average='weighted')
        recall = recall_score(self.all_labels, self.all_predictions, average='weighted')
        f1 = f1_score(self.all_labels, self.all_predictions, average='weighted')

        # Per-class metrics
        precision_per_class = precision_score(self.all_labels, self.all_predictions, average=None)
        recall_per_class = recall_score(self.all_labels, self.all_predictions, average=None)
        f1_per_class = f1_score(self.all_labels, self.all_predictions, average=None)

        # AUC-ROC (for binary classification)
        try:
            auc_roc = roc_auc_score(self.all_labels, self.all_probabilities[:, 1])
        except Exception:
            auc_roc = None

        # Confusion matrix
        cm = confusion_matrix(self.all_labels, self.all_predictions)

        # Classification report
        report = classification_report(
            self.all_labels,
            self.all_predictions,
            target_names=list(self.class_names.values()),
            output_dict=True
        )

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': cm.tolist(),
            'per_class': {
                self.class_names[i]: {
                    'precision': precision_per_class[i],
                    'recall': recall_per_class[i],
                    'f1_score': f1_per_class[i]
                }
                for i in range(len(self.class_names))
            },
            'classification_report': report,
            'total_samples': len(self.all_labels)
        }

        return metrics

    def print_metrics(self, metrics: Dict) -> None:
        """Print metrics in a formatted way"""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        print(f"\nTotal Samples: {metrics['total_samples']}")
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")

        if 'precision' in metrics:
            print(f"  Precision: {metrics['precision']*100:.2f}%")
            print(f"  Recall:    {metrics['recall']*100:.2f}%")
            print(f"  F1-Score:  {metrics['f1_score']*100:.2f}%")

            if metrics['auc_roc']:
                print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

        print(f"\nPer-Class Metrics:")
        if 'per_class' in metrics:
            for class_name, class_metrics in metrics['per_class'].items():
                print(f"\n  {class_name}:")
                print(f"    Precision: {class_metrics['precision']*100:.2f}%")
                print(f"    Recall:    {class_metrics['recall']*100:.2f}%")
                print(f"    F1-Score:  {class_metrics['f1_score']*100:.2f}%")
        elif 'per_class_accuracy' in metrics:
            for class_name, acc in metrics['per_class_accuracy'].items():
                print(f"  {class_name}: {acc*100:.2f}%")

        if 'confusion_matrix' in metrics:
            print(f"\nConfusion Matrix:")
            cm = np.array(metrics['confusion_matrix'])
            print(f"  {'':>15} {'Pred Real':>12} {'Pred AI':>12}")
            print(f"  {'True Real':>15} {cm[0,0]:>12} {cm[0,1]:>12}")
            print(f"  {'True AI':>15} {cm[1,0]:>12} {cm[1,1]:>12}")

        print("\n" + "=" * 60)

    def plot_confusion_matrix(self, metrics: Dict, save_path: Optional[str] = None) -> None:
        """Plot confusion matrix"""
        if not PLOTTING_AVAILABLE or 'confusion_matrix' not in metrics:
            return

        cm = np.array(metrics['confusion_matrix'])
        class_names = list(self.class_names.values())

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Evaluator] Confusion matrix saved to: {save_path}")

        plt.show()

    def plot_roc_curve(self, save_path: Optional[str] = None) -> None:
        """Plot ROC curve"""
        if not PLOTTING_AVAILABLE or not SKLEARN_AVAILABLE:
            return

        try:
            fpr, tpr, _ = roc_curve(self.all_labels, self.all_probabilities[:, 1])
            auc = roc_auc_score(self.all_labels, self.all_probabilities[:, 1])

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"[Evaluator] ROC curve saved to: {save_path}")

            plt.show()

        except Exception as e:
            print(f"[Evaluator] Could not plot ROC curve: {e}")

    def get_misclassified_samples(self, dataset, num_samples: int = 10) -> List[Dict]:
        """Get misclassified samples for analysis"""
        misclassified_indices = np.where(self.all_predictions != self.all_labels)[0]

        samples = []
        for idx in misclassified_indices[:num_samples]:
            sample_info = dataset.get_sample_info(idx)
            sample_info['predicted'] = self.class_names[self.all_predictions[idx]]
            sample_info['confidence'] = float(self.all_probabilities[idx].max())
            samples.append(sample_info)

        return samples

    def evaluate(self, save_results: bool = True, output_dir: str = LOGS_DIR) -> Dict:
        """
        Run complete evaluation pipeline.

        Args:
            save_results: Whether to save results to files
            output_dir: Directory to save results

        Returns:
            Dictionary of metrics
        """
        # Run inference
        self.run_inference()

        # Calculate metrics
        metrics = self.calculate_metrics()

        # Print metrics
        self.print_metrics(metrics)

        # Save and plot
        if save_results:
            os.makedirs(output_dir, exist_ok=True)

            # Save metrics JSON
            metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')

            # Convert numpy types to Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                return obj

            serializable_metrics = {
                k: convert_to_serializable(v) if not isinstance(v, dict)
                else {kk: convert_to_serializable(vv) for kk, vv in v.items()}
                for k, v in metrics.items()
            }

            with open(metrics_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=2, default=str)
            print(f"\n[Evaluator] Metrics saved to: {metrics_path}")

            # Plot confusion matrix
            cm_path = os.path.join(output_dir, 'confusion_matrix.png')
            self.plot_confusion_matrix(metrics, save_path=cm_path)

            # Plot ROC curve
            roc_path = os.path.join(output_dir, 'roc_curve.png')
            self.plot_roc_curve(save_path=roc_path)

        return metrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate DeepFake Face Detector')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (uses best_model.pth if not specified)')
    parser.add_argument('--output-dir', type=str, default=LOGS_DIR,
                        help='Directory to save evaluation results')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save evaluation results')

    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()

    # Create directories
    create_directories()

    # Get device
    device = get_device()

    # Load model
    checkpoint_path = args.checkpoint or os.path.join(MODEL_DIR, 'best_model.pth')

    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        print("[Error] Please train a model first using: python train.py")
        return

    model, checkpoint = load_model_for_inference(checkpoint_path, device)

    # Prepare data
    print("\n[Main] Preparing test data...")
    datasets, dataloaders = prepare_data()

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=dataloaders['test'],
        device=device
    )

    # Run evaluation
    metrics = evaluator.evaluate(
        save_results=not args.no_save,
        output_dir=args.output_dir
    )

    # Get misclassified samples
    print("\n[Main] Sample misclassified images:")
    misclassified = evaluator.get_misclassified_samples(datasets['test'], num_samples=5)
    for i, sample in enumerate(misclassified, 1):
        print(f"  {i}. {os.path.basename(sample['path'])}")
        print(f"     True: {sample['label_name']}, Predicted: {sample['predicted']}")
        print(f"     Confidence: {sample['confidence']*100:.1f}%")

    print("\n[Main] Evaluation complete!")


if __name__ == "__main__":
    main()
