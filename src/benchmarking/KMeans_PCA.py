import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score

with open("src/benchmarking/cell_types_subset_1.txt", "r") as f:
    cell_types = [line.strip() for line in f]

embed_data = np.load("models/pca_subset_embeddings.npy")

for k in [2, 3, 8, 16, 32]:
    for j in [11, 30, 50]:
        avg_accuracy = []
        nmi_scores = []
        ari_scores = []
        for i in range(30):
            data = embed_data[:, :k]
            kmeans = KMeans(n_clusters=j, random_state=i).fit(data)
            preds = kmeans.labels_

            # Match predicted cluster labels to true labels
            cluster_to_true_label = {}
            for cluster in range(j):
                cluster_indices = [i for i, label in enumerate(preds) if label == cluster]
                if cluster_indices:
                    true_labels_in_cluster = [cell_types[i] for i in cluster_indices]
                    most_common_label = max(set(true_labels_in_cluster), key=true_labels_in_cluster.count)
                    cluster_to_true_label[cluster] = most_common_label

            # Map predicted labels to true labels
            mapped_preds = [cluster_to_true_label[label] for label in preds]

            avg_accuracy.append(accuracy_score(cell_types, mapped_preds))
            nmi_scores.append(normalized_mutual_info_score(cell_types, preds))
            ari_scores.append(adjusted_rand_score(cell_types, preds))

        print(f"KMeans (n_clusters={j}): Accuracy: {np.mean(avg_accuracy)} - Std: {np.std(avg_accuracy)}, NMI: {np.mean(nmi_scores)} - Std: {np.std(nmi_scores)}, ARI: {np.mean(ari_scores)} - Std: {np.std(ari_scores)}")