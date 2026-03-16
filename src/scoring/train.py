"""Training loop with Leave-One-Out cross-validation for the ST-GCN regressor.

Usage
-----
    uv run python -m src.scoring.train
    uv run python -m src.scoring.train --poses-dir output/poses_3d --epochs 300

LOOCV strategy:
  - 22 folds: each fold holds out 1 video for testing.
  - Inside each fold, 1 additional video is held out from the remaining 21 for
    early stopping (the video with the highest index in the sorted training list).
  - Augmentation is applied to the 20-video training set.
  - Score normalization: min-max to [0,1] computed on the training set only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.scoring.dataset import PoseDataset, scan_snowboard_men
from src.scoring.model import STGCNRegressor


def _minmax_scale(
    scores: list[float],
) -> tuple[np.ndarray, float, float]:
    """Scale scores to [0, 1]. Returns (scaled, min, max)."""
    y = np.array(scores, dtype=np.float32)
    y_min, y_max = y.min(), y.max()
    return (y - y_min) / (y_max - y_min + 1e-8), float(y_min), float(y_max)


def _unscale(y_scaled: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    return y_scaled * (y_max - y_min + 1e-8) + y_min


def _train_one_fold(
    train_records: list,
    val_records: list,
    *,
    epochs: int,
    crop_len: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    verbose: bool = False,
) -> tuple[STGCNRegressor, float, float]:
    """Train a model for one LOOCV fold.

    Returns
    -------
    model : STGCNRegressor  best checkpoint
    y_min : float  score normalization lower bound
    y_max : float  score normalization upper bound
    """
    all_train_scores = [s for _, s, _ in train_records]
    y_scaled, y_min, y_max = _minmax_scale(all_train_scores)

    # Build training dataset with augmentation
    train_ds = PoseDataset(
        [(s, sc, p) for s, sc, p in train_records],
        crop_len=crop_len,
        augment=True,
    )
    # Overwrite scores with normalized values for training
    train_ds._scores = [float(v) for v in y_scaled]

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)

    # Build validation dataset (no augmentation, no crop)
    val_ds = PoseDataset(
        [(s, sc, p) for s, sc, p in val_records],
        crop_len=None,
        augment=False,
    )
    val_scores_raw = [s for _, s, _ in val_records]

    model = STGCNRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.SmoothL1Loss(beta=0.1)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # ── Training ──────────────────────────────────────────────────
        # train_ds._scores already holds normalized values; targets are in [0,1]
        model.train()
        for poses, targets in train_loader:
            poses = poses.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(poses)
            loss = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for i in range(len(val_ds)):
                poses_seq, score_raw, _ = val_ds.get_full_sequence(i)
                poses_seq = poses_seq.unsqueeze(0).to(device)
                pred_norm = model(poses_seq).item()
                pred_raw = _unscale(np.array([pred_norm]), y_min, y_max)[0]
                val_loss_total += abs(pred_raw - val_scores_raw[i])
        val_loss = val_loss_total / len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch} (best val MAE={best_val_loss:.2f})")
                break

        if verbose and epoch % 50 == 0:
            print(f"    Epoch {epoch:4d}/{epochs} | val MAE={val_loss:.2f}")

    model.load_state_dict(best_state)
    return model, y_min, y_max


def run_loocv(
    poses_dir: str | Path = "output/poses_3d",
    *,
    epochs: int = 300,
    crop_len: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    patience: int = 50,
    device_str: str = "cpu",
    verbose: bool = True,
) -> list[dict]:
    """Run full Leave-One-Out CV and return per-fold predictions.

    Returns
    -------
    list[dict]  Each dict has keys: stem, score_true, score_pred, fold_idx.
    """
    device = torch.device(device_str)
    records = scan_snowboard_men(poses_dir)
    N = len(records)
    if N < 3:
        raise RuntimeError(f"Need at least 3 videos, found {N} in {poses_dir}")

    print(f"LOOCV: {N} videos, {epochs} epochs, device={device_str}")
    results: list[dict] = []

    for fold_idx, (test_stem, test_score, test_path) in enumerate(records):
        t0 = time.time()
        # All other videos for training
        train_val = [r for r in records if r[0] != test_stem]

        # Hold out last video (highest index) from train_val for early stopping
        val_records = [train_val[-1]]
        train_records = train_val[:-1]

        if verbose:
            print(f"\nFold {fold_idx + 1}/{N}: test={test_stem} (score={test_score})")

        model, y_min, y_max = _train_one_fold(
            train_records,
            val_records,
            epochs=epochs,
            crop_len=crop_len,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            device=device,
            verbose=verbose,
        )

        # Inference on test video (full sequence)
        test_ds = PoseDataset([(test_stem, test_score, test_path)], crop_len=None, augment=False)
        model.eval()
        with torch.no_grad():
            poses_seq, _, _ = test_ds.get_full_sequence(0)
            poses_seq = poses_seq.unsqueeze(0).to(device)
            pred_norm = model(poses_seq).item()
            pred_raw = _unscale(np.array([pred_norm]), y_min, y_max)[0]

        elapsed = time.time() - t0
        if verbose:
            print(f"  → pred={pred_raw:.2f}  true={test_score:.2f}  ({elapsed:.0f}s)")

        results.append({
            "fold_idx": fold_idx,
            "stem": test_stem,
            "score_true": float(test_score),
            "score_pred": float(pred_raw),
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="ST-GCN LOOCV training")
    parser.add_argument("--poses-dir", default="output/poses_3d")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--crop-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", default="cpu", help="cpu | mps | cuda:0")
    parser.add_argument("--out", default="output/scoring/stgcn_loocv.json")
    args = parser.parse_args()

    results = run_loocv(
        poses_dir=args.poses_dir,
        epochs=args.epochs,
        crop_len=args.crop_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device_str=args.device,
        verbose=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved predictions → {out_path}")


if __name__ == "__main__":
    main()
