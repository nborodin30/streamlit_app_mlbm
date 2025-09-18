"""
ML in Biomedicine Practical Course
"""

from __future__ import annotations

import os

from omegaconf import OmegaConf, DictConfig
import torch
import hydra
import logging
import pandas as pd 

from ml.utils import set_determinism, log_factory,save_json
from data import SequenceWindowDataset, build_sequence_dataloaders
from model import ExonIntronCNN
from ml.train import fit
from data_preprocessing import preprocess_windows, load_fasta, balance_dataset


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:  # noqa: D401
    """Execute the end-to-end scientific ML pipeline.

    This function orchestrates the entire workflow, from configuration loading
    and determinism setup to data processing, model training, and artifact
    persistence. It adheres to a strict, fail-fast philosophy.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra configuration object, composed from YAML files and command-line
        overrides. It contains all settings for the run.

    """
    logger = logging.getLogger(__name__)
    logger.info("Loaded config:\n" + OmegaConf.to_yaml(cfg))

    # 1. Determinism and environment setup
    set_determinism(cfg.train.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.train.device)

    # 2. Data preprocessing: from FASTA + mask to balanced CSV of windows
    # Load and preprocess data
    ref = load_fasta(cfg.sequence.reference_fasta)
    mask = load_fasta(cfg.sequence.mask_fasta)
    windows = preprocess_windows(ref, mask, cfg.sequence.window_size)
    balanced_windows = balance_dataset(windows, cfg.train.seed)

    seq_ds = SequenceWindowDataset(balanced_windows, window_size=cfg.sequence.window_size)
    # 3. DataLoader setup with chromosome-based splitting
    train_loader, val_loader, test_loader = build_sequence_dataloaders(
        seq_ds,
        batch_size=cfg.optim.batch_size,
        val_fraction=cfg.data.val_size,
        test_fraction=cfg.data.test_size,
        seed=cfg.train.seed,
        val_chroms=list(cfg.data.val_chroms) if cfg.data.val_chroms else None,
        test_chroms=list(cfg.data.test_chroms) if cfg.data.test_chroms else None,
    )
    # Pre‑training diagnostics
    logger.info(
        "Dataset windows: total=%d | train=%d | val=%d | test=%d",
        len(seq_ds),
        len(train_loader.dataset),  # type: ignore[arg-type]
        len(val_loader.dataset),  # type: ignore[arg-type]
        len(test_loader.dataset),  # type: ignore[arg-type]
    )
    logger.info(
        "Config (core): window_size=%d label_policy=%s batch_size=%d lr=%.3g weight_decay=%.3g epochs=%d device=%s",
        cfg.sequence.window_size,
        cfg.sequence.label_policy,
        cfg.optim.batch_size,
        cfg.optim.lr,
        cfg.optim.weight_decay,
        cfg.optim.epochs,
        device,
    )
    logger.info(
        "Estimated steps/epoch: %d (len(train_loader)) | total updates ~ %d",
        len(train_loader),
        len(train_loader) * cfg.optim.epochs,
    )

    in_channels = 5  # only nucleotide one-hot channels (mask used solely for labels)
    model = ExonIntronCNN(
        in_channels=in_channels,
        conv_channels=list(cfg.model.conv_channels),
        kernel_size=cfg.model.kernel_size,
        dropout=cfg.model.dropout,
        num_classes=2,
    ).to(device)

    norm_stats = {
        "mode": "sequence",
        "window_size": cfg.sequence.window_size,
        "label_policy": cfg.sequence.label_policy,
    }
    label_map = {"intron": 0, "exon": 1}

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info("Model architecture:\n%s", repr(model))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    logger.info("Optimizer: Adam(lr=%.3g, weight_decay=%.3g)", cfg.optim.lr, cfg.optim.weight_decay)
    logger.info("Loss: CrossEntropyLoss | Starting training loop ...")

    # Set up logging and execute training
    log_fn = log_factory(logger)

    results = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=cfg.optim.epochs,
        device=device,
        log_fn=log_fn,
        log_interval=cfg.train.log_interval,
        patience=cfg.train.patience,
    )
    logger.info("Final results: " + ", ".join(f"{k}={v:.4f}" for k, v in results.items()))

    # 9. Persist artifacts
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore[attr-defined]
    model_path = os.path.join(output_dir, "model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "results": results,
            "normalization": norm_stats,
            "label_map": label_map,
        },
        model_path,
    )
    logger.info(f"Saved model checkpoint to {model_path}")

    # Also store normalization stats and results separately for easy inspection
    save_json(norm_stats, os.path.join(output_dir, "normalization_stats.json"))
    save_json(results, os.path.join(output_dir, "results.json"))

    logger.info(f"Saved normalization stats and results to {output_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()  # type: ignore
