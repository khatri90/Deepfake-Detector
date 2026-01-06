"""
Training Script for DeepFake Face Detector

Features:
- Automatic mixed precision training (if GPU available)
- Learning rate scheduling with warmup
- Early stopping
- Checkpoint saving (best & last)
- Progress tracking with tqdm
- Backbone unfreezing for fine-tuning
"""
import os
import sys
import time
import argparse
from datetime import datetime
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

# Try importing tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[Train] tqdm not available, using basic progress output")

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, MODEL_DIR, LOGS_DIR,
    UNFREEZE_AT_EPOCH, MODEL_NAME, create_directories
)
from model import create_model
from preprocessing import prepare_data
from utils import (
    set_seed, get_device, EarlyStopping, MetricsTracker,
    save_checkpoint, print_model_summary, get_timestamp
)


class Trainer:
    """
    Trainer class for DeepFake Face Detector.

    Handles the complete training loop with:
    - Mixed precision training
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Metric tracking
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = DEVICE,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        num_epochs: int = NUM_EPOCHS,
        unfreeze_at_epoch: int = UNFREEZE_AT_EPOCH,
        patience: int = EARLY_STOPPING_PATIENCE,
        use_amp: bool = True,
        checkpoint_dir: str = MODEL_DIR,
        log_dir: str = LOGS_DIR
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )

        # Mixed precision training (only for CUDA)
        self.use_amp = use_amp and device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, mode='min')

        # Metrics tracker
        self.metrics = MetricsTracker()

        # Best metrics
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')

        # Training state
        self.current_epoch = 0
        self.backbone_unfrozen = False

        print(f"\n[Trainer] Initialized")
        print(f"  Device: {device}")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Unfreeze backbone at epoch: {unfreeze_at_epoch}")

    def _train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar
        if TQDM_AVAILABLE:
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        else:
            pbar = self.train_loader
            print(f"  Training...", end=" ", flush=True)

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass (with mixed precision if enabled)
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            if TQDM_AVAILABLE:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

        avg_loss = running_loss / len(self.train_loader)
        accuracy = correct / total

        if not TQDM_AVAILABLE:
            print(f"Loss: {avg_loss:.4f}, Acc: {100.*accuracy:.2f}%")

        return avg_loss, accuracy

    @torch.no_grad()
    def _validate_epoch(self) -> Tuple[float, float]:
        """
        Validate for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar
        if TQDM_AVAILABLE:
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
        else:
            pbar = self.val_loader
            print(f"  Validating...", end=" ", flush=True)

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if TQDM_AVAILABLE:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

        avg_loss = running_loss / len(self.val_loader)
        accuracy = correct / total

        if not TQDM_AVAILABLE:
            print(f"Loss: {avg_loss:.4f}, Acc: {100.*accuracy:.2f}%")

        return avg_loss, accuracy

    def _unfreeze_backbone(self):
        """Unfreeze backbone and reinitialize optimizer"""
        if not self.backbone_unfrozen:
            print(f"\n[Trainer] Unfreezing backbone at epoch {self.current_epoch + 1}")
            self.model.unfreeze_backbone()

            # Reinitialize optimizer with all parameters
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=LEARNING_RATE * 0.1,  # Lower LR for fine-tuning
                weight_decay=WEIGHT_DECAY
            )

            self.backbone_unfrozen = True
            print(f"[Trainer] Learning rate reduced to {LEARNING_RATE * 0.1} for fine-tuning")

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'model_name': getattr(self.model, 'model_name', MODEL_NAME),
            'metrics_history': self.metrics.history
        }

        # Save last checkpoint
        last_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pth')
        torch.save(checkpoint, last_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"[Trainer] Best model saved! Val Acc: {self.best_val_acc:.4f}")

    def train(self) -> Dict:
        """
        Run the complete training loop.

        Returns:
            Dictionary with training results
        """
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"{'='*60}")

            # Check if we should unfreeze backbone
            if epoch == self.unfreeze_at_epoch:
                self._unfreeze_backbone()

            # Train
            train_loss, train_acc = self._train_epoch()

            # Validate
            val_loss, val_acc = self._validate_epoch()

            # Update learning rate scheduler
            self.scheduler.step(val_loss)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update metrics
            self.metrics.update(train_loss, train_acc, val_loss, val_acc, current_lr)

            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss

            # Save checkpoint
            self._save_checkpoint(is_best=is_best)

            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\n[Summary] Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"[Summary] Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
            print(f"[Summary] LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            print(f"[Summary] Best Val Acc: {self.best_val_acc*100:.2f}%")

            # Early stopping check
            if self.early_stopping(val_loss, epoch):
                print(f"\n[Trainer] Early stopping triggered!")
                break

        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc*100:.2f}%")
        print(f"Best model saved to: {os.path.join(self.checkpoint_dir, 'best_model.pth')}")

        # Save metrics
        metrics_path = os.path.join(self.log_dir, f'training_metrics_{get_timestamp()}.json')
        self.metrics.save(metrics_path)

        # Plot metrics (if matplotlib available)
        try:
            plot_path = os.path.join(self.log_dir, f'training_plot_{get_timestamp()}.png')
            self.metrics.plot(save_path=plot_path)
        except Exception as e:
            print(f"[Trainer] Could not save plot: {e}")

        return {
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'final_epoch': self.current_epoch + 1,
            'total_time': total_time
        }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train DeepFake Face Detector')

    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (uses config default if not specified)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directories
    create_directories()

    # Get device
    device = get_device()

    # Prepare data
    print("\n[Main] Preparing data...")
    datasets, dataloaders = prepare_data()

    # Create model
    print("\n[Main] Creating model...")
    model = create_model()
    print_model_summary(model)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        device=device,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        use_amp=not args.no_amp
    )

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\n[Main] Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.current_epoch = checkpoint['epoch'] + 1
            trainer.best_val_acc = checkpoint.get('best_val_acc', 0)
            print(f"[Main] Resumed from epoch {trainer.current_epoch}")
        else:
            print(f"[Warning] Checkpoint not found: {args.resume}")

    # Train
    results = trainer.train()

    print("\n[Main] Training finished!")
    print(f"[Main] Results: {results}")

    return results


if __name__ == "__main__":
    main()
