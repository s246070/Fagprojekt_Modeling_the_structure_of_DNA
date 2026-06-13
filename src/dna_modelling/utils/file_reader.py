import os


def read_file(name):
    losses = []
    aucs = []
    f1_scores = []
    pr_aucs = []
    with open(f"data/{name}.out", "r") as f:
        lines = f.readlines()
        if lines[:5] == "Epoch":
            loss, auc, f1_score, pr_auc = lines[0].split("|")
            loss = float(loss.split(":")[1].strip())
            auc = float(auc.split(":")[1].strip())
            f1_score = float(f1_score.split(":")[1].strip())
            pr_auc = float(pr_auc.split(":")[1].strip())
            epochs.append(epoch)
            losses.append(loss)
            aucs.append(auc)
            f1_scores.append(f1_score)
            pr_aucs.append(pr_auc)
    
    with open(f"data/{name}.csv", "w") as f:
        f.write("Loss,AUC,F1-Score,PR-AUC\n")
        for i in range(len(losses)):
            f.write(f"{losses[i]},{aucs[i]},{f1_scores[i]},{pr_aucs[i]}\n")