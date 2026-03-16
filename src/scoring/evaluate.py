"""Evaluation metrics and reporting for the scoring module.

Metrics:
  - Spearman's rank correlation (ρ): measures ranking quality
  - Relative L2 distance: mean(|y_pred - y_true| / y_true)

Usage
-----
    # Run full pipeline: train ST-GCN + baseline, then compare
    uv run python -m src.scoring.evaluate

    # Load existing predictions
    uv run python -m src.scoring.evaluate --stgcn-preds output/scoring/stgcn_loocv.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scipy.stats import spearmanr
    _SCIPY = True
except ImportError:
    _SCIPY = False


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    if _SCIPY:
        rho, _ = spearmanr(y_true, y_pred)
        return float(rho)
    # Manual implementation
    n = len(y_true)
    rank_true = _rank(y_true)
    rank_pred = _rank(y_pred)
    d2 = ((rank_true - rank_pred) ** 2).sum()
    return float(1.0 - 6.0 * d2 / (n * (n * n - 1)))


def _rank(x: np.ndarray) -> np.ndarray:
    """1-based ranks (average for ties)."""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks


def relative_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean relative absolute error: mean(|y_pred - y_true| / y_true)."""
    return float(np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def print_results_table(
    stems: list[str],
    scores_true: list[float],
    scores_pred: list[float],
    label: str = "Model",
) -> None:
    """Print per-fold prediction table and aggregate metrics."""
    y_true = np.array(scores_true)
    y_pred = np.array(scores_pred)
    errors = np.abs(y_pred - y_true)

    rho = spearman_rho(y_true, y_pred)
    rel_l2 = relative_l2(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\n{'='*70}")
    print(f"  {label} — LOOCV Results")
    print(f"{'='*70}")
    print(f"  {'Video stem':<45}  {'True':>6}  {'Pred':>6}  {'|Err|':>6}")
    print(f"  {'-'*45}  {'------':>6}  {'------':>6}  {'------':>6}")
    for stem, true, pred, err in zip(stems, scores_true, scores_pred, errors):
        short = stem[:45]
        print(f"  {short:<45}  {true:>6.2f}  {pred:>6.2f}  {err:>6.2f}")
    print(f"  {'-'*65}")
    print(f"  Spearman ρ:        {rho:+.3f}")
    print(f"  Relative L2:       {rel_l2:.3f}")
    print(f"  MAE:               {mae:.2f}")
    print(f"{'='*70}")


def compute_metrics(
    scores_true: list[float], scores_pred: list[float]
) -> dict[str, float]:
    y_true = np.array(scores_true)
    y_pred = np.array(scores_pred)
    return {
        "spearman_rho": spearman_rho(y_true, y_pred),
        "relative_l2": relative_l2(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "n": len(y_true),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate scoring module")
    parser.add_argument("--poses-dir", default="output/poses_3d")
    parser.add_argument("--stgcn-preds", default=None, help="Path to existing LOOCV predictions JSON")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="output/scoring")
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── ST-GCN ────────────────────────────────────────────────────────
    stgcn_results: list[dict] | None = None

    if args.stgcn_preds and Path(args.stgcn_preds).exists():
        with open(args.stgcn_preds) as f:
            stgcn_results = json.load(f)
        print(f"Loaded ST-GCN predictions from {args.stgcn_preds}")
    else:
        print("Training ST-GCN (LOOCV)...")
        from src.scoring.train import run_loocv
        stgcn_results = run_loocv(
            poses_dir=args.poses_dir,
            epochs=args.epochs,
            device_str=args.device,
            verbose=True,
        )
        stgcn_path = out_dir / "stgcn_loocv.json"
        with open(stgcn_path, "w") as f:
            json.dump(stgcn_results, f, indent=2)
        print(f"Saved → {stgcn_path}")

    stems = [r["stem"] for r in stgcn_results]
    y_true = [r["score_true"] for r in stgcn_results]
    y_pred_stgcn = [r["score_pred"] for r in stgcn_results]

    print_results_table(stems, y_true, y_pred_stgcn, label="ST-GCN")
    stgcn_metrics = compute_metrics(y_true, y_pred_stgcn)

    # ── Baseline ──────────────────────────────────────────────────────
    if not args.no_baseline:
        print("\nComputing hand-crafted feature baseline (Ridge LOOCV)...")
        from src.scoring.dataset import scan_snowboard_men, load_poses
        from src.scoring.baseline import RidgeBaseline, extract_features

        records = scan_snowboard_men(args.poses_dir)
        # Only include videos that appear in stgcn_results (same set)
        stgcn_stems = {r["stem"] for r in stgcn_results}
        records = [r for r in records if r[0] in stgcn_stems]

        features = [extract_features(load_poses(path)) for _, _, path in records]
        bl_scores = [score for _, score, _ in records]

        baseline = RidgeBaseline(alpha=1.0)
        bl_preds = baseline.fit_predict_loocv(features, bl_scores)
        bl_stems = [r[0] for r in records]

        print_results_table(bl_stems, bl_scores, bl_preds.tolist(), label="Ridge Baseline")
        bl_metrics = compute_metrics(bl_scores, bl_preds.tolist())

        # Save baseline predictions
        bl_out = [
            {"stem": s, "score_true": t, "score_pred": float(p)}
            for s, t, p in zip(bl_stems, bl_scores, bl_preds)
        ]
        bl_path = out_dir / "baseline_loocv.json"
        with open(bl_path, "w") as f:
            json.dump(bl_out, f, indent=2)
        print(f"Saved → {bl_path}")

        # ── Comparison ────────────────────────────────────────────────
        print(f"\n{'='*50}")
        print("  Comparison Summary")
        print(f"{'='*50}")
        print(f"  {'Metric':<20}  {'ST-GCN':>8}  {'Ridge':>8}")
        print(f"  {'-'*40}")
        print(f"  {'Spearman ρ':<20}  {stgcn_metrics['spearman_rho']:>+8.3f}  {bl_metrics['spearman_rho']:>+8.3f}")
        print(f"  {'Relative L2':<20}  {stgcn_metrics['relative_l2']:>8.3f}  {bl_metrics['relative_l2']:>8.3f}")
        print(f"  {'MAE':<20}  {stgcn_metrics['mae']:>8.2f}  {bl_metrics['mae']:>8.2f}")
        print(f"{'='*50}")

    # Save summary
    summary = {"stgcn": stgcn_metrics}
    if not args.no_baseline:
        summary["ridge_baseline"] = bl_metrics
    summary_path = out_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
