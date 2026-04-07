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
    # Implementation for creating a test set
    targets = []


    for cell in range(Aij.shape[0]):
        for peak in range(Aij.shape[1]):
            if Aij[cell, peak] == 1:
                # Randomly decide to remove this connection based on the percentage
                if torch.rand(1).item() < percentage:
                    Aij[cell, peak] = 0  # Remove the connection
                    # Store the removed connection for evaluation
                    targets.append((cell, peak))

    return Aij, targets


def evaluate(model, Aij, targets, increment=0.001):
    """

    Evaluates the accuracy of predictions against the removed connections.

    Args:
        model: The trained LDM model.
        targets: A list of tuples (cell_index, peak_index) representing the removed connections.

    Returns:
        A list of tuples (FPR, TPR) for different thresholds, which can be used to plot an AUROC curve.
    """
    i = 0.0

    # Make AUROC curve data
    auroc_data = []
    while i <= 1.0:
        predictions = model.probabilities() >= i
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for cell in range(Aij.shape[0]):
            for peak in range(Aij.shape[1]):
                if (cell, peak) in targets:  # Actually connected
                    if predictions[cell, peak] == 1:  # Predicted as connected
                        true_positives += 1
                    else:  # Predicted as not connected
                        false_negatives += 1
                elif Aij[cell, peak] == 0:  # Not actually connected
                    if predictions[cell, peak] == 1:  # Predicted as connected
                        false_positives += 1
                    else:  # Predicted as not connected
                        true_negatives += 1

        # Calculate TPR and FPR for the current threshold
        tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0

        auroc_data.append((fpr, tpr))
        i += increment

    # Calculate the area under the curve (AUC) using the trapezoidal rule
    auc = 0.0
    for j in range(1, len(auroc_data)):
        fpr_prev, tpr_prev = auroc_data[j - 1]
        fpr_curr, tpr_curr = auroc_data[j]
        auc += (fpr_curr - fpr_prev) * (tpr_prev + tpr_curr) / 2

    return auc, auroc_data