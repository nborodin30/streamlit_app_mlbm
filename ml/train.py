"""Minimal training helpers: accuracy, one training/eval epoch, and a fit loop
that keeps the best validation model and reports final test metrics."""

from __future__ import annotations

from typing import Dict, Callable, Any, Sequence
import time
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef
import warnings


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return classification accuracy (0-1) for logits (N,C) vs targets (N)."""
    assert logits.ndim == 2, "Logits should be a 2D tensor."
    assert targets.ndim == 1, "Targets should be a 1D tensor."
    assert logits.shape[0] == targets.shape[0], "Batch sizes must match."
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.numel()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    log_fn: Callable[[Dict[str, float]], None],
    log_interval: int,
) -> Dict[str, float]:
    """One training pass over `loader`; returns {loss, acc}."""
    model.train()
    losses = []
    accs = []
    all_preds = []
    all_targets = []
    for batch_idx, (batch_x, batch_y) in enumerate(loader, start=1):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Forward pass
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Compute batch metrics
        batch_loss = loss.item()
        batch_acc = accuracy(logits.detach(), batch_y)
        preds = torch.argmax(logits.detach(), dim=1)

        y_true = batch_y.cpu().numpy().tolist()
        y_pred = preds.cpu().numpy().tolist()

        if len(set(y_true) | set(y_pred)) > 1:
            batch_mcc = matthews_corrcoef(y_true, y_pred)
        else:
            batch_mcc = 0.0

        # Record loss, acc, and preds for overall metrics
        losses.append(batch_loss)
        accs.append(batch_acc)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(batch_y.cpu().numpy().tolist())

        # Log every log_interval batches
        if batch_idx % log_interval == 0:
            log_fn({"batch": batch_idx, "loss": batch_loss, "acc": batch_acc, "mcc": batch_mcc})

    # Compute Matthews correlation coefficient
    mcc = matthews_corrcoef(all_targets, all_preds)
    return {"loss": float(np.mean(losses)), "acc": float(np.mean(accs)), "mcc": float(mcc)}


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> Dict[str, float]:
    """Evaluate (no grad) over `loader`; returns {loss, acc}."""
    model.eval()
    losses = []
    accs = []
    all_preds = []
    all_targets = []
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        # Record loss and accuracy
        losses.append(loss.item())
        accs.append(accuracy(logits, batch_y))
        # Record predictions and targets for MCC
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(batch_y.cpu().numpy().tolist())

    # Compute Matthews correlation coefficient
    mcc = matthews_corrcoef(all_targets, all_preds) if (all_targets or all_preds) else 0.0
    return {"loss": float(np.mean(losses)), "acc": float(np.mean(accs)), "mcc": float(mcc)}


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
    log_fn: Callable[[Dict[str, Any]], None],
    log_interval: int,
    patience: int = 5,
) -> Dict[str, float]:
    """Train for `epochs`, track best val acc (restoring its weights), then test.
    Returns dict(best_val_acc, test_loss, test_acc)."""
    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, log_fn, log_interval
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        elapsed_time = time.time() - start_time

        log_obj = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_mcc": train_metrics["mcc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_mcc": val_metrics["mcc"],
            "time_sec": elapsed_time,
        }
        log_fn(log_obj)

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            # Use a deepcopy to ensure the state is fully independent
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0  # Reset patience counter on improvement
        else:
            patience_counter += 1
            if patience_counter >= patience:
                warnings.warn(f"Early stopping at epoch {epoch} due to no improvement in val_acc.")
                break

    assert best_state is not None, "Training loop failed to produce a best model state."
    model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, loss_fn, device)
    # Include MCC for test set
    return {
        "best_val_acc": best_val_acc,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "test_mcc": test_metrics["mcc"],
    }
