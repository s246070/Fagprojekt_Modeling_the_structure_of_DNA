import torch

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

    Aij[remove_mask] = 0
    removed_indices = remove_mask.nonzero(as_tuple=False)
    targets = [tuple(index) for index in removed_indices.cpu().tolist()]

    n_targets = removed_indices.shape[0]
    if n_targets == 0:
        return Aij, targets, []

    # Sample negatives from original zero entries without replacement.
    zero_candidates = (~connected_mask).flatten().nonzero(as_tuple=False).squeeze(1)
    if zero_candidates.numel() < n_targets:
        raise ValueError("Not enough zero entries to sample target_zeros without replacement")

    sampled_ids = zero_candidates[torch.randperm(zero_candidates.numel(), device=Aij.device)[:n_targets]]
    row_idx = sampled_ids // Aij.shape[1]
    col_idx = sampled_ids % Aij.shape[1]
    target_zeros = [tuple(index) for index in torch.stack((row_idx, col_idx), dim=1).cpu().tolist()]

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

    # Make AUROC curve data
    auroc_data = []
    f1_score = 0
    thresholds = torch.arange(0.0, 1.0 + increment, increment, device=probabilities.device)
    
    for threshold in thresholds.tolist():
        if threshold > 1.0:
            continue

        tp = sum(probabilities[cell_idx, peak_idx] >= threshold for cell_idx, peak_idx in targets)
        fp = sum(probabilities[cell_idx, peak_idx] >= threshold for cell_idx, peak_idx in target_zeros)
        fn = sum(probabilities[cell_idx, peak_idx] < threshold for cell_idx, peak_idx in targets)
        tn = sum(probabilities[cell_idx, peak_idx] < threshold for cell_idx, peak_idx in target_zeros)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        auroc_data.append((fpr, tpr))
        
        if precision + recall > 0 and 2 * (precision * recall) / (precision + recall) > f1_score:
            f1_score = 2 * (precision * recall) / (precision + recall)


    # Calculate the area under the curve (AUC) using the trapezoidal rule
    if not auroc_data:
        return 0.0, auroc_data, f1_score

    curve_tensor = torch.tensor(auroc_data, dtype=torch.float32)
    sorted_indices = torch.argsort(curve_tensor[:, 0])
    fpr_sorted = curve_tensor[sorted_indices, 0]
    tpr_sorted = curve_tensor[sorted_indices, 1]
    auc = torch.trapz(tpr_sorted, fpr_sorted).item()

    return auc, auroc_data, f1_score
