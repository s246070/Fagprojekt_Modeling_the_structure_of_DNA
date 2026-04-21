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
    return Aij, targets


def evaluate(model, Aij, targets, increment=0.001):
    """
    Evaluates the accuracy of predictions against the removed connections.

    Args:
        model: The trained LDM model.
        Aij: The modified cell x peak accessibility matrix with some connections removed (tensor).
        targets: A list of tuples (cell_index, peak_index) representing the removed connections.
        increment: Amount of steps between 0 and 1 for calculating TPR and FPR at different thresholds (default: 0.001).

    Returns:
        A list of tuples (FPR, TPR) for different thresholds, which can be used to plot an AUROC curve.
    """
    if increment <= 0:
        raise ValueError("increment must be greater than 0")

    target_mask = torch.zeros_like(Aij, dtype=torch.bool)
    if targets:
        target_indices = torch.as_tensor(targets, device=Aij.device, dtype=torch.long)
        target_mask[target_indices[:, 0], target_indices[:, 1]] = True
    
    negative_mask = (Aij == 0) & (~target_mask)
    probabilities = model.probabilities()

    # Make AUROC curve data
    auroc_data = []
    f1_data = []
    thresholds = torch.arange(0.0, 1.0 + increment, increment, device=probabilities.device)
    
    for threshold in thresholds.tolist():
        if threshold > 1.0:
            continue

        predictions = probabilities >= threshold
        true_positives = (predictions & target_mask).sum().item()
        false_negatives = ((~predictions) & target_mask).sum().item()
        false_positives = (predictions & negative_mask).sum().item()
        true_negatives = ((~predictions) & negative_mask).sum().item()

        # Calculate TPR and FPR for the current threshold
        tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0

        auroc_data.append((fpr, tpr))

        # Calculate F1 score for the current threshold
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = tpr
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_data.append((threshold, f1))

    # Calculate the area under the curve (AUC) using the trapezoidal rule
    if not auroc_data:
        return 0.0, auroc_data, f1_data

    curve_tensor = torch.tensor(auroc_data, dtype=torch.float32)
    sorted_indices = torch.argsort(curve_tensor[:, 0])
    fpr_sorted = curve_tensor[sorted_indices, 0]
    tpr_sorted = curve_tensor[sorted_indices, 1]
    auc = torch.trapz(tpr_sorted, fpr_sorted).item()

    return auc, auroc_data, f1_data