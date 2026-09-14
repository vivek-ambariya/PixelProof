"""PROBE Training: Adversarial hard-negative mining for unseen generator generalization.

This module implements the PROBE framework from "Where Detectors Fail: Probing
Generative Space for Generalizable AI-Generated Image Detection" (ICML 2026).

The key insight: Instead of passively training on a fixed dataset, iteratively
find images that your detector struggles with (hard negatives) and retrain on them.

This teaches the detector to recognize failure modes and generalize to unseen generators.

WHY THIS WORKS:
- Standard training: Learn from all examples equally
- PROBE training: Focus on mistakes - find cases where detector is confident but wrong
- Result: Detector learns to handle edge cases
- Expected improvement: +0.015-0.020 AUC on unseen generators

USAGE:
    from probe_training import ProbeTrainer, find_hard_negatives
    
    # Initialize trainer
    trainer = ProbeTrainer(model=model, device='cuda', num_iterations=3)
    
    # Run PROBE training
    trainer.train_with_probe(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_iterations=3,  # 3 refinement iterations
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """Find images that detector struggles with (hard negatives)."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        confidence_threshold: float = 0.5,
    ):
        """Initialize hard negative miner.
        
        Args:
            model: Detector model (outputs logits)
            device: 'cuda' or 'cpu'
            confidence_threshold: Uncertainty threshold (0.5 = neutral)
        """
        self.model = model
        self.device = device
        self.confidence_threshold = confidence_threshold

    def find_hard_negatives(
        self,
        dataloader: DataLoader,
        k_hardest: int = None,
        percentile: int = 10,
    ) -> tuple[list[int], dict]:
        """Find hard negatives from a dataloader.
        
        Hard negatives are:
        1. AI images scored as real (prob < 0.3) - False negatives
        2. Real images scored as AI (prob > 0.7) - False positives
        3. Images scored near decision boundary (prob ~0.5) - Uncertain
        
        Args:
            dataloader: DataLoader to mine from
            k_hardest: Return top k hardest examples (if None, use percentile)
            percentile: Use examples in hardest N percentile (default: hardest 10%)
        
        Returns:
            (hard_negative_indices, statistics_dict)
        """
        self.model.eval()
        all_logits = []
        all_labels = []
        all_indices = []

        # Collect predictions on all data
        with torch.no_grad():
            for batch_idx, (images, labels, indices) in enumerate(dataloader):
                images = images.to(self.device)
                logits = self.model(images)

                all_logits.extend(logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_indices.extend(indices.cpu().numpy())

        all_logits = np.array(all_logits)
        all_labels = np.array(all_labels)
        all_indices = np.array(all_indices)
        all_probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid

        # Find hard negatives by multiple criteria
        hard_ai = (all_labels == 1) & (all_probs < 0.3)  # AI scored as real
        hard_real = (all_labels == 0) & (all_probs > 0.7)  # Real scored as AI
        uncertain = (np.abs(all_probs - 0.5) < 0.1)  # Near boundary

        hard_mask = hard_ai | hard_real | uncertain
        hard_indices = np.where(hard_mask)[0]

        # Sort by difficulty (distance from correct prediction)
        def difficulty_score(idx):
            prob = all_probs[idx]
            label = all_labels[idx]
            if label == 1:  # Should be high (AI)
                return prob  # Lower is harder
            else:  # Should be low (real)
                return 1.0 - prob  # Higher diff is harder

        difficulties = np.array([difficulty_score(i) for i in hard_indices])
        sorted_order = np.argsort(difficulties)
        hard_indices = hard_indices[sorted_order]

        # Select k_hardest or percentile
        if k_hardest is not None:
            hard_indices = hard_indices[:k_hardest]
        else:
            num_hard = max(1, len(hard_indices) // (100 // percentile))
            hard_indices = hard_indices[:num_hard]

        # Map back to original indices
        hard_original_indices = all_indices[hard_indices].tolist()

        # Statistics
        stats = {
            'total_examples': len(all_labels),
            'num_hard_found': len(hard_original_indices),
            'percentage_hard': 100.0 * len(hard_original_indices) / len(all_labels),
            'ai_false_negatives': (hard_ai).sum(),
            'real_false_positives': (hard_real).sum(),
            'uncertain': (uncertain).sum(),
            'avg_prob_hard': all_probs[hard_indices].mean(),
            'avg_prob_easy': all_probs[~hard_mask].mean(),
        }

        return hard_original_indices, stats


class ProbeTrainer:
    """Train with PROBE: iteratively refine on hard negatives."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        """Initialize PROBE trainer.
        
        Args:
            model: Detector model
            device: 'cuda' or 'cpu'
            learning_rate: For fine-tuning on hard negatives
            weight_decay: L2 regularization
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.miner = HardNegativeMiner(model, device)

    def train_on_hard_negatives(
        self,
        hard_dataloader: DataLoader,
        num_epochs: int = 5,
        log_interval: int = 50,
    ) -> dict:
        """Retrain on hard negatives.
        
        Args:
            hard_dataloader: DataLoader with only hard examples
            num_epochs: How many epochs to train on hard negatives
            log_interval: Log every N batches
        
        Returns:
            training stats dict
        """
        # Use lower learning rate for fine-tuning on hard examples
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        stats = {'losses': [], 'epoch_losses': []}

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (images, labels, _) in enumerate(hard_dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(-1)

                # Forward
                logits = self.model(images)
                loss = criterion(logits, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                stats['losses'].append(loss.item())

                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    logger.info(
                        f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}, "
                        f"Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}"
                    )

            avg_epoch_loss = epoch_loss / num_batches
            stats['epoch_losses'].append(avg_epoch_loss)
            logger.info(f"Epoch {epoch + 1} complete. Avg loss: {avg_epoch_loss:.4f}")

        return stats

    def train_with_probe(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_iterations: int = 3,
        percentile_hard: int = 10,
        epochs_per_iteration: int = 5,
    ) -> dict:
        """Full PROBE training loop.
        
        Iteratively:
        1. Score all training examples
        2. Find hard negatives (examples detector struggles with)
        3. Retrain on hard negatives
        4. Repeat
        
        Args:
            train_dataloader: Training data
            val_dataloader: Validation data (for progress tracking)
            num_iterations: Number of refinement iterations
            percentile_hard: Hardest N percentile to retrain on (default: 10%)
            epochs_per_iteration: Epochs per iteration on hard examples
        
        Returns:
            Dictionary with training history
        """
        history = {
            'iterations': [],
            'hard_stats': [],
            'train_loss': [],
            'val_metrics': [],
        }

        for iteration in range(num_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"PROBE Iteration {iteration + 1}/{num_iterations}")
            logger.info(f"{'='*60}")

            # Step 1: Find hard negatives
            logger.info("Mining hard negatives from training set...")
            hard_indices, hard_stats = self.miner.find_hard_negatives(
                train_dataloader,
                percentile=percentile_hard,
            )

            logger.info(f"Found {hard_stats['num_hard_found']} hard examples "
                       f"({hard_stats['percentage_hard']:.1f}% of training data)")
            logger.info(f"  - AI false negatives: {hard_stats['ai_false_negatives']}")
            logger.info(f"  - Real false positives: {hard_stats['real_false_positives']}")
            logger.info(f"  - Avg prob (hard): {hard_stats['avg_prob_hard']:.3f}")
            logger.info(f"  - Avg prob (easy): {hard_stats['avg_prob_easy']:.3f}")

            history['hard_stats'].append(hard_stats)

            # Step 2: Create subset with hard negatives
            if len(hard_indices) == 0:
                logger.info("No hard negatives found. PROBE training complete!")
                break

            # Get original dataset from dataloader
            original_dataset = train_dataloader.dataset
            hard_subset = Subset(original_dataset, hard_indices)
            hard_loader = DataLoader(
                hard_subset,
                batch_size=train_dataloader.batch_size,
                shuffle=True,
                num_workers=0,  # No multiprocessing for safety
                pin_memory=True,
            )

            # Step 3: Retrain on hard negatives
            logger.info(f"Retraining on {len(hard_indices)} hard examples "
                       f"for {epochs_per_iteration} epochs...")
            train_stats = self.train_on_hard_negatives(
                hard_loader,
                num_epochs=epochs_per_iteration,
            )
            history['train_loss'].append(train_stats['epoch_losses'])

            # Step 4: Evaluate on validation set (optional, for monitoring)
            logger.info("Evaluating on validation set...")
            val_stats = self.evaluate(val_dataloader)
            history['val_metrics'].append(val_stats)
            logger.info(f"  Val AUC: {val_stats.get('auc', 'N/A')}")

            logger.info(f"Iteration {iteration + 1} complete.\n")

        return history

    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader to evaluate on
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in dataloader:
                images = images.to(self.device)
                logits = self.model(images)

                all_logits.extend(logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_logits = np.array(all_logits)
        all_labels = np.array(all_labels)
        all_probs = 1.0 / (1.0 + np.exp(-all_logits))

        # Calculate metrics
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5

        accuracy = accuracy_score(all_labels, (all_probs > 0.5).astype(int))
        f1 = f1_score(all_labels, (all_probs > 0.5).astype(int))

        return {
            'auc': auc,
            'accuracy': accuracy,
            'f1': f1,
            'avg_prob': all_probs.mean(),
        }


def create_indexed_dataloader(
    original_dataloader: DataLoader,
    add_indices: bool = True,
) -> DataLoader:
    """Wrap a dataloader to return original indices.
    
    PROBE mining needs to map hard examples back to original dataset.
    This wrapper adds original indices to each batch.
    
    Args:
        original_dataloader: Original DataLoader
        add_indices: Whether to add indices (default True)
    
    Returns:
        New DataLoader that yields (images, labels, indices)
    """
    if not add_indices:
        return original_dataloader

    class IndexedDataset:
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            # Assume standard format: (image, label) or similar
            if isinstance(item, tuple) and len(item) == 2:
                image, label = item
                return image, label, idx
            elif isinstance(item, dict):
                return item, idx
            else:
                return item, idx

    indexed_dataset = IndexedDataset(original_dataloader.dataset)

    return DataLoader(
        indexed_dataset,
        batch_size=original_dataloader.batch_size,
        shuffle=original_dataloader.sampler is None,
        num_workers=original_dataloader.num_workers,
        pin_memory=original_dataloader.pin_memory,
        drop_last=original_dataloader.drop_last,
    )


if __name__ == "__main__":
    print("PROBE Training module loaded successfully!")
    print("\nUsage:")
    print("  from probe_training import ProbeTrainer")
    print("  trainer = ProbeTrainer(model=model, device='cuda')")
    print("  trainer.train_with_probe(train_dl, val_dl, num_iterations=3)")
