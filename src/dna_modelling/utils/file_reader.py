from pathlib import Path


def read_file(name):
    epochs = []
    losses = []
    aucs = []
    f1_scores = []
    pr_aucs = []

    input_path = Path(f"data/{name}.out")
    output_path = Path(f"data/{name}.csv")

    with input_path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if "Epoch" not in line or "Loss:" not in line:
                continue

            parts = [part.strip() for part in line.split("|")]
            epoch_part = next((part for part in parts if part.startswith("Epoch ")), None)
            loss_part = next((part for part in parts if part.startswith("Loss:")), None)
            auc_part = next((part for part in parts if part.startswith("AUC (100%):")), None)
            f1_part = next((part for part in parts if part.startswith("F1 Score:")), None)
            pr_auc_part = next((part for part in parts if part.startswith("PR AUC:")), None)

            if (
                epoch_part is None
                or loss_part is None
                or auc_part is None
                or f1_part is None
                or pr_auc_part is None
            ):
                continue

            epochs.append(int(epoch_part.split()[1].split("/")[0]))
            losses.append(float(loss_part.split(":")[1].strip()))
            aucs.append(float(auc_part.split(":")[1].strip()))
            f1_scores.append(float(f1_part.split(":")[1].strip()))
            pr_aucs.append(float(pr_auc_part.split(":")[1].strip()))

    with output_path.open("w", encoding="utf-8") as file_handle:
        file_handle.write("Epoch,Loss,AUC,F1-Score,PR-AUC\n")
        for epoch, loss, auc, f1_score, pr_auc in zip(epochs, losses, aucs, f1_scores, pr_aucs):
            file_handle.write(f"{epoch},{loss},{auc},{f1_score},{pr_auc}\n")