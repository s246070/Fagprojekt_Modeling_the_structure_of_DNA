import os
import matplotlib.pyplot as plt
import numpy as np

loss = {}
auc_values = {}
pr_auc_values = {}
f1_values = {}

for i in [2, 3, 8, 16, 32]:
    for j in range(1, 11):
        name = f"ldm_ls{i}_epoch{j * 100}_blocks100_index1"
        with open(f"results/{name}.csv", "r") as f:
            data = f.read().strip().split(",")

        if i not in loss:
            loss[i] = []
            auc_values[i] = []
            f1_values[i] = []
            pr_auc_values[i] = []
        loss[i].append(float(data[0]))
        auc_values[i].append(float(data[1]))
        f1_values[i].append(float(data[2]))
        pr_auc_values[i].append(float(data[3]))

plt.figure(figsize=(12, 8))
for i in loss:
    plt.plot([j * 100 for j in range(1, 11)], loss[i], marker='o', label=f'ls_dim={i}')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("plots/loss_vs_epochs_batching.png", dpi=300)
plt.clf()


for i in auc_values:
    plt.plot([j * 100 for j in range(1, 11)], auc_values[i], marker='o', label=f'ls_dim={i}')
plt.xlabel("Epochs")
plt.ylabel("AUC Values")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("plots/auc_vs_epochs_batching.png", dpi=300)
plt.clf()

for i in f1_values:
    plt.plot([j * 100 for j in range(1, 11)], f1_values[i], marker='o', label=f'ls_dim={i}')
plt.xlabel("Epochs")
plt.ylabel("F1 Score")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("plots/f1_score_vs_epochs_batching.png", dpi=300)
plt.clf()

for i in pr_auc_values:
    plt.plot([j * 100 for j in range(1, 11)], pr_auc_values[i], marker='o', label=f'ls_dim={i}')
plt.xlabel("Epochs")
plt.ylabel("PR AUC Values")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("plots/pr_auc_vs_epochs_batching.png", dpi=300)
plt.clf()