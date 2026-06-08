import numpy as np
import torch
from typing import Tuple, List, Dict, Any

from ..dna_modelling.evaluate import make_test_set


def _to_dense_numpy(X) -> np.ndarray:
    """Convert an AnnData X matrix (sparse or dense) to a dense numpy array of 0/1 values."""
    try:
        # Load backed / lazy data into memory before conversion
        if hasattr(X, "__getitem__") and not isinstance(X, (list, tuple, np.ndarray)):
            try:
                X = X[:]
            except Exception:
                pass

        import scipy.sparse as sp

        if sp.issparse(X):
            arr = X.toarray()
        else:
            arr = np.asarray(X)
    except Exception:
        arr = np.asarray(X)

    # binarize: PeakVI models presence/absence
    arr = (arr > 0).astype(np.int64)
    return arr


def evaluate_peakvi(
    model,
    adata,
    test_fraction: float = 0.1,
    increment: float = 0.001,
    use_z_mean: bool = True,
    batch_size: int = 128,
    targets: List[Tuple[int, int]] = None,
    target_zeros: List[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Evaluate a trained `scvi.model.PEAKVI` model.

    Steps:
    - Extract binary accessibility matrix from `adata.X`.
    - Use pre-computed targets and target_zeros if provided, otherwise create them.
    - Use `model.get_normalized_accessibility(..., return_numpy=True)` to get probability estimates.
    - Compute AUROC, AUPR (area under precision-recall), and max F1 over thresholds.

    Args:
        model: Trained PeakVI model.
        adata: AnnData object with accessibility data.
        test_fraction: Fraction of positives to use as test (only if targets not provided).
        increment: Threshold increment for ROC/PR computation.
        use_z_mean: Whether to use mean latent representation.
        batch_size: Batch size for model inference.
        targets: Pre-computed list of (cell, region) tuples for positive test set.
        target_zeros: Pre-computed list of (cell, region) tuples for negative test set.

    Returns a dict with keys: `auroc`, `aupr`, `f1_max`, `auroc_curve`, `pr_curve`.
    """

    # get probabilities from the model (cells x regions)
    probs = model.get_normalized_accessibility(
        adata=adata,
        n_samples_overall=None,
        use_z_mean=use_z_mean,
        batch_size=batch_size,
        return_numpy=True,
    )
    probs = np.asarray(probs)

    if len(targets) == 0:
        raise ValueError("No positive targets were sampled; decrease `test_fraction` or check input matrix.")

    # build score and label arrays
    tgt_rows, tgt_cols = zip(*targets)
    neg_rows, neg_cols = zip(*target_zeros)

    y_true = np.concatenate([np.ones(len(tgt_rows)), np.zeros(len(neg_rows))])
    y_scores = np.concatenate([probs[np.array(tgt_rows), np.array(tgt_cols)], probs[np.array(neg_rows), np.array(neg_cols)]])

    # thresholds
    thresholds = np.arange(0.0, 1.0 + increment, increment)

    # vectorized TP/FP/FN/TN computation
    scores_pos = y_scores[: len(tgt_rows)][:, None]
    scores_neg = y_scores[len(tgt_rows) :][:, None]
    thr = thresholds[None, :]

    tp = (scores_pos >= thr).sum(axis=0).astype(float)
    fp = (scores_neg >= thr).sum(axis=0).astype(float)
    fn = (scores_pos < thr).sum(axis=0).astype(float)
    tn = (scores_neg < thr).sum(axis=0).astype(float)

    eps = 1e-8
    tpr = tp / (tp + fn + eps)
    fpr = fp / (fp + tn + eps)
    precision = tp / (tp + fp + eps)
    recall = tpr

    f1 = 2 * (precision * recall) / (precision + recall + eps)
    f1_max = float(np.nanmax(f1))

    # AUROC: sort by fpr
    sort_idx = np.argsort(fpr)
    fpr_sorted = fpr[sort_idx]
    tpr_sorted = tpr[sort_idx]
    auroc = float(np.trapz(tpr_sorted, fpr_sorted))

    # AUPR: sort by recall
    sort_pr = np.argsort(recall)
    recall_sorted = recall[sort_pr]
    precision_sorted = precision[sort_pr]
    aupr = float(np.trapz(precision_sorted, recall_sorted))

    auroc_curve = list(zip(fpr.tolist(), tpr.tolist()))
    pr_curve = list(zip(precision.tolist(), recall.tolist()))

    return {
        "auroc": auroc,
        "aupr": aupr,
        "f1_max": f1_max,
        "auroc_curve": auroc_curve,
        "pr_curve": pr_curve,
    }


if __name__ == "__main__":
    print("This module provides `evaluate_peakvi(model, adata, ...)`.")
