import torch
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
import numpy as np

with open("src/benchmarking/cell_types.txt", "r") as f:
    cell_types = [line.strip() for line in f]

data = torch.load("models/peakvi_latent_2d.pth")

for i in [11, 20, 30]:
    avg_accuracy = []
    nmi_scores = []
    ari_scores = []
    for j in range(30):
        kmeans = KMeans(n_clusters=i, random_state=j).fit(data)
        preds = kmeans.labels_

        # Match predicted cluster labels to true labels
        cluster_to_true_label = {}
        for cluster in range(i):
            cluster_indices = [j for j, label in enumerate(preds) if label == cluster]
            if cluster_indices:
                true_labels_in_cluster = [cell_types[j] for j in cluster_indices]
                most_common_label = max(set(true_labels_in_cluster), key=true_labels_in_cluster.count)
                cluster_to_true_label[cluster] = most_common_label

        # Map predicted labels to true labels
        mapped_preds = [cluster_to_true_label[label] for label in preds]

        avg_accuracy.append(accuracy_score(cell_types, mapped_preds))
        nmi_scores.append(normalized_mutual_info_score(cell_types, mapped_preds))
        ari_scores.append(adjusted_rand_score(cell_types, mapped_preds))

    print(f"KMeans (n_clusters={i}): Accuracy: {np.mean(avg_accuracy)} - Std: {np.std(avg_accuracy)}, NMI: {np.mean(nmi_scores)} - Std: {np.std(nmi_scores)}, ARI: {np.mean(ari_scores)} - Std: {np.std(ari_scores)}")