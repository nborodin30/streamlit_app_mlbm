"""Model definitions for intron/exon classification.

This module provides a compact 1D convolutional neural network tailored for
genomic window classification (exon vs intron). The model is intentionally
minimal, prioritising:

* Determinism & reproducibility (no implicit randomness beyond dropout)
* Clear shape invariants (assertions guard all critical assumptions)
* Extensibility (layers kept in an ordered list so they can be swapped easily)

The network expects an input tensor of shape: ``(batch, channels, seq_len)``
where channels = 5 nucleotide one‑hot (A,C,G,T,N) + 1 mask channel (accessible
region indicator) = 6 by default. The exact channel count is passed explicitly
to avoid hidden coupling with preprocessing.

Architecture (configurable via ``config.yaml``):
    [Conv1d -> ReLU -> (Dropout)] x N  -> GlobalAvgPool (across sequence) -> Linear

Global average pooling eliminates any dependence on sequence length at the
classification head, provided the conv stack preserves sequence dimension.
"""

from __future__ import annotations

from typing import List
import torch
from torch import nn


class ExonIntronCNN(nn.Module):
    """Small 1D CNN for exon vs intron classification.

    Parameters
    ----------
    in_channels : int
        Number of input channels (5 nucleotide one‑hot + optional mask channel).
    conv_channels : List[int]
        Output channels for each Conv1d block. Length defines depth.
    kernel_size : int
        Kernel size for all conv layers (must be odd to preserve center alignment).
    dropout : float
        Dropout probability applied after each ReLU (0 -> disabled).
    num_classes : int
        Number of target classes (2 for intron/exon).

    Notes
    -----
    * All convolutions use padding = kernel_size//2 to keep sequence length stable.
    * Global average pooling removes sequence length at the classifier head.
    * Assertions ensure configuration sanity (fail fast philosophy).
    """

    def __init__(
        self,
        in_channels: int,
        conv_channels: List[int],
        kernel_size: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        assert in_channels > 0, "in_channels must be positive"
        assert num_classes >= 2, "Need at least two classes"
        assert kernel_size % 2 == 1, "kernel_size should be odd for symmetric padding"
        assert 0.0 <= dropout <= 1.0, "dropout outside [0,1]"
        assert len(conv_channels) > 0, "Provide at least one conv layer"

        layers: List[nn.Module] = []
        prev_c = in_channels
        padding = kernel_size // 2
        for out_c in conv_channels:
            layers.append(nn.Conv1d(prev_c, out_c, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_c = out_c
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_c, num_classes)
        #increase the number of linaer layers to 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        assert x.ndim == 3, "Expected input shape (batch, channels, seq_len)"
        feats = self.feature_extractor(x)  # (B, C, L)
        # Global average pool across sequence length
        pooled = feats.mean(dim=2)  # (B, C)
        return self.classifier(pooled)


__all__ = ["ExonIntronCNN"]
