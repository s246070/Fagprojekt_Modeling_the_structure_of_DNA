import torch
from datetime import datetime

def make_test_set(Aij, percentage=0.1):
    """
    Creates a test set by randomly removing a percentage of the connections in the Aij matrix.
    
    Args:
        Aij: The original cell x peak accessibility matrix (tensor).
        percentage: The percentage of connections to remove for the test set.
    
    Returns:
        A modified Aij matrix with some connections removed, and a list of the removed connections (targets).
    """
    if not 0 <= percentage <= 1:
        raise ValueError("percentage must be between 0 and 1")

    connected_mask = Aij == 1
    sampled_mask = torch.rand(Aij.shape, device=Aij.device) < percentage
    remove_mask = connected_mask & sampled_mask
    print(f"begin removing connections{datetime.now()}", flush=True)
    Aij[remove_mask] = 0
    removed_indices = remove_mask.nonzero(as_tuple=True)
    print(f"begin target thing{datetime.now()}", flush=True)
    targets = list(zip(removed_indices[0].cpu().tolist(), removed_indices[1].cpu().tolist()))

    print(f"Doing n targets{datetime.now()}", flush=True)
    n_targets = removed_indices[0].shape[0]
    if n_targets == 0:
        return Aij, targets, []

    print(f"Removed some connections, now sampling negatives{datetime.now()}", flush=True)
    # Sample negatives from original zero entries without replacement.
    zero_candidates = (~connected_mask).flatten().nonzero(as_tuple=False).squeeze(1)
    if zero_candidates.numel() < n_targets:
        raise ValueError("Not enough zero entries to sample target_zeros without replacement")

    print(f"begin sampling target_zeros{datetime.now()}", flush=True)
    sampled_ids = zero_candidates[torch.randperm(zero_candidates.numel(), device=Aij.device)[:n_targets]]
    print(f"begin calculating target_zeros{datetime.now()}", flush=True)
    row_idx = sampled_ids // Aij.shape[1]
    col_idx = sampled_ids % Aij.shape[1]
    print(f"Begin target_zeros{datetime.now()}", flush=True)
    target_zeros = list(zip(row_idx.cpu().tolist(), col_idx.cpu().tolist()))

    return Aij, targets, target_zeros


def validate(model, Aij, targets, target_zeros, increment=0.001):
    """
    Evaluates the accuracy of predictions against the removed connections.

    Args:
        model: The trained LDM model.
        Aij: The modified cell x peak accessibility matrix with some connections removed (tensor).
        targets: A list of tuples (cell_index, peak_index) representing the removed connections.
        target_zeros: A list of tuples (cell_index, peak_index) representing the negative connections.
        increment: Amount of steps between 0 and 1 for calculating TPR and FPR at different thresholds (default: 0.001).

    Returns:
        A list of tuples (FPR, TPR) for different thresholds, which can be used to plot an AUROC curve.
    """
    if increment <= 0:
        raise ValueError("increment must be greater than 0")

    probabilities = model.probabilities()

    # Extract probabilities for targets and target_zeros in batch
    targets_array = torch.tensor(targets, dtype=torch.long, device=probabilities.device)
    target_zeros_array = torch.tensor(target_zeros, dtype=torch.long, device=probabilities.device)

    target_probs = probabilities[targets_array[:, 0], targets_array[:, 1]]
    target_zero_probs = probabilities[target_zeros_array[:, 0], target_zeros_array[:, 1]]

    # Create thresholds and filter out anything > 1.0
    thresholds = torch.arange(0.0, 1.0 + increment, increment, device=probabilities.device)
    thresholds = thresholds[thresholds <= 1.0]

    # Vectorized threshold comparison using broadcasting
    target_probs_expanded = target_probs.unsqueeze(1)
    target_zero_probs_expanded = target_zero_probs.unsqueeze(1)
    thresholds_expanded = thresholds.unsqueeze(0)

    # Calculate TP, FP, FN, TN for all thresholds at once
    tp = (target_probs_expanded >= thresholds_expanded).sum(dim=0)
    fp = (target_zero_probs_expanded >= thresholds_expanded).sum(dim=0)
    fn = (target_probs_expanded < thresholds_expanded).sum(dim=0)
    tn = (target_zero_probs_expanded < thresholds_expanded).sum(dim=0)

    # Calculate TPR, FPR, Precision, Recall with numerical stability
    tpr = tp / (tp + fn).clamp(min=1)
    fpr = fp / (fp + tn).clamp(min=1)
    precision = tp / (tp + fp).clamp(min=1)
    recall = tpr

    # Calculate max F1 score across all thresholds
    recall_micro = (tp + tn) / (tp + tn + fn + fp).clamp(min=1e-8)
    precision_micro = (tp + tn) / (tp + tn + fn + fp).clamp(min=1e-8)
    f1_micro_score = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro).clamp(min=1e-8)
    f1_micro_score = f1_micro_score.max().item()

    # Create AUROC data
    auroc_data = [(fpr[i].item(), tpr[i].item()) for i in range(len(thresholds))]
    pr_curve_data = [(precision[i].item(), recall[i].item()) for i in range(len(thresholds))]

    # Calculate the area under the curve (AUC) using the trapezoidal rule
    if not auroc_data:
        return 0.0, [], f1_micro_score, 0.0, pr_curve_data

    curve_tensor = torch.tensor(auroc_data, dtype=torch.float32)
    sorted_indices = torch.argsort(curve_tensor[:, 0])
    fpr_sorted = curve_tensor[sorted_indices, 0]
    tpr_sorted = curve_tensor[sorted_indices, 1]
    auc = torch.trapz(tpr_sorted, fpr_sorted).item()
    
    # Sort PR curve by recall for proper AUC calculation
    pr_curve_tensor = torch.tensor(pr_curve_data, dtype=torch.float32)
    recall_vals = pr_curve_tensor[:, 1]
    precision_vals = pr_curve_tensor[:, 0]
    sorted_pr_indices = torch.argsort(recall_vals)
    recall_sorted = recall_vals[sorted_pr_indices]
    precision_sorted = precision_vals[sorted_pr_indices]
    pr_auc = torch.trapz(precision_sorted, recall_sorted).item()

    return auc, auroc_data, f1_micro_score, pr_auc, pr_curve_data

def validate_pairwise(model, target_ones, target_zeros, increment=0.001):
    """
    Evaluate held-out cell-peak indices pairwise and return metrics.

    target_ones and target_zeros should contain held-out cell-peak indices.

    Expected format:
        target_ones = tensor/list/array of shape (n, 2)
        target_zeros = tensor/list/array of shape (m, 2)

    Returns:
        auc: float (AUROC)
        auroc_data: list of (FPR, TPR) tuples for thresholds
        f1_micro_score: float (maximum micro F1 across thresholds)
        pr_auc: float (area under precision-recall curve)
        pr_curve_data: list of (precision, recall) tuples for thresholds
    """

    if increment <= 0:
        raise ValueError("increment must be greater than 0")

    was_training = model.training
    model.eval()

    with torch.no_grad():
        target_ones = torch.as_tensor(target_ones, device=model.device).long()
        target_zeros = torch.as_tensor(target_zeros, device=model.device).long()

        pos_cell_idx = target_ones[:, 0]
        pos_feature_idx = target_ones[:, 1]

        neg_cell_idx = target_zeros[:, 0]
        neg_feature_idx = target_zeros[:, 1]

        pos_logits = model(pos_cell_idx, pos_feature_idx)
        neg_logits = model(neg_cell_idx, neg_feature_idx)

        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([
            torch.ones_like(pos_logits),
            torch.zeros_like(neg_logits)
        ])

        probs = torch.sigmoid(logits)

    # Restore original training/eval state
    model.train(was_training)

    probs = probs.detach().cpu()
    labels = labels.detach().cpu()

    # Separate positive and negative probabilities
    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]

    n_pos = pos_probs.shape[0]
    n_neg = neg_probs.shape[0]
    if n_pos == 0 or n_neg == 0:
        # Not enough data to compute ROC/PR
        return 0.0, [], 0.0, 0.0, []

    # thresholds
    thresholds = torch.arange(0.0, 1.0 + increment, increment)
    thresholds = thresholds[thresholds <= 1.0]

    pos_exp = pos_probs.unsqueeze(1)
    neg_exp = neg_probs.unsqueeze(1)
    thresh_exp = thresholds.unsqueeze(0)

    tp = (pos_exp >= thresh_exp).sum(dim=0)
    fp = (neg_exp >= thresh_exp).sum(dim=0)
    fn = (pos_exp < thresh_exp).sum(dim=0)
    tn = (neg_exp < thresh_exp).sum(dim=0)

    tpr = tp / (tp + fn).clamp(min=1)
    fpr = fp / (fp + tn).clamp(min=1)
    precision = tp / (tp + fp).clamp(min=1)
    recall = tpr

    # Micro-like F1 calculation (kept consistent with validate())
    recall_micro = (tp + tn) / (tp + tn + fn + fp).clamp(min=1e-8)
    precision_micro = recall_micro
    f1_micro_score = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro).clamp(min=1e-8)
    f1_micro_score = f1_micro_score.max().item()

    auroc_data = [(fpr[i].item(), tpr[i].item()) for i in range(len(thresholds))]
    pr_curve_data = [(precision[i].item(), recall[i].item()) for i in range(len(thresholds))]

    # Compute AUC (AUROC)
    curve_tensor = torch.tensor(auroc_data, dtype=torch.float32)
    sorted_indices = torch.argsort(curve_tensor[:, 0])
    fpr_sorted = curve_tensor[sorted_indices, 0]
    tpr_sorted = curve_tensor[sorted_indices, 1]
    auc = torch.trapz(tpr_sorted, fpr_sorted).item()

    # Compute PR AUC
    pr_curve_tensor = torch.tensor(pr_curve_data, dtype=torch.float32)
    recall_vals = pr_curve_tensor[:, 1]
    precision_vals = pr_curve_tensor[:, 0]
    sorted_pr_indices = torch.argsort(recall_vals)
    recall_sorted = recall_vals[sorted_pr_indices]
    precision_sorted = precision_vals[sorted_pr_indices]
    pr_auc = torch.trapz(precision_sorted, recall_sorted).item()

    return auc, auroc_data, f1_micro_score, pr_auc, pr_curve_data