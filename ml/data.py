"""Sequence data utilities for intron/exon classification using preprocessed CSV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

@dataclass
class SequenceMeta:
    """Optional metadata container (reserved for future extension)."""
    window_size: int
    channels: int

NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

def _one_hot_encode(seq: str) -> np.ndarray:
    """One-hot encode an uppercase DNA sequence into shape (5, L)."""
    L = len(seq)
    arr = np.zeros((5, L), dtype=np.float32)
    for i, base in enumerate(seq):
        idx = NUC_TO_IDX.get(base, 4)  # unknown -> N
        arr[idx, i] = 1.0
    return arr

class SequenceWindowDataset(Dataset):
    """Dataset loading preprocessed windows from CSV."""
    def __init__(self, data: List[Dict], window_size: int = 127):
        """Initialize with a list of dictionaries containing 'sequence' and 'label'."""
        self.window_size = window_size
        # Validate data structure
        if not data or not isinstance(data, list):
            raise ValueError("data must be a non-empty list of dictionaries")
        for d in data:
            if not isinstance(d, dict) or 'sequence' not in d or 'label' not in d or 'chrom' not in d:
                raise KeyError("Each dictionary must contain 'sequence', 'label', and 'chrom' keys")
            if len(d['sequence']) != window_size:
                raise ValueError(f"Sequence length {len(d['sequence'])} does not match window_size {window_size}")
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data[idx]
        seq = row['sequence']
        label = row['label']
        one_hot = _one_hot_encode(seq)
        return torch.from_numpy(one_hot), torch.tensor(label, dtype=torch.long)

def build_sequence_dataloaders(
    dataset: SequenceWindowDataset,
    batch_size: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    val_chroms: Optional[List[str]] = None,
    test_chroms: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test loaders using chromosome-based splitting."""
    chrom_for_index = [d['chrom'] for d in dataset.data]  # CHANGED: List comprehension instead of DataFrame access
    all_chroms = sorted(set(chrom_for_index))
    if val_chroms:
        for c in val_chroms:
            assert c in all_chroms, f"val chromosome {c} not present"
    if test_chroms:
        for c in test_chroms:
            assert c in all_chroms, f"test chromosome {c} not present"
    if (val_chroms and test_chroms) and set(val_chroms) & set(test_chroms):
        raise AssertionError("val_chroms and test_chroms overlap")
    if not val_chroms or not test_chroms:
        rng = random.Random(seed)
        chroms_shuffled = all_chroms[:]
        rng.shuffle(chroms_shuffled)
        n_chroms = len(chroms_shuffled)
        val_k = max(1, int(round(n_chroms * val_fraction)))
        test_k = max(1, int(round(n_chroms * test_fraction)))
        if val_k + test_k >= n_chroms:
            test_k = max(1, test_k - 1)
        if not val_chroms:
            val_chroms = chroms_shuffled[:val_k]
        if not test_chroms:
            test_chroms = chroms_shuffled[val_k : val_k + test_k]
    train_chroms = [c for c in all_chroms if c not in set(val_chroms) | set(test_chroms)]
    assert train_chroms, "No chromosomes left for training after split"
    
    train_idx = [i for i, c in enumerate(chrom_for_index) if c in train_chroms]
    val_idx = [i for i, c in enumerate(chrom_for_index) if c in val_chroms]
    test_idx = [i for i, c in enumerate(chrom_for_index) if c in test_chroms]
    assert train_idx and val_idx and test_idx, "Empty split subset encountered"

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
    )

__all__ = [
    "SequenceWindowDataset",
    "build_sequence_dataloaders",
    "SequenceMeta",
]