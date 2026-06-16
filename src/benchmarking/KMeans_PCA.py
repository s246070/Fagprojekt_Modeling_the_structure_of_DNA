import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score

with open("src/benchmarking/cell_types_subset_1.txt", "r") as f:
    cell_types = [line.strip() for line in f]

embed_data = np.load("models/pca_subset_embeddings.npy")

for k in [2, 3, 8, 16, 32]:
    data = embed_data[:, :k]
    for i in [11, 20, 30]:
        kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
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

        print(f"KMeans (n_clusters={i}): Accuracy: {accuracy_score(cell_types, mapped_preds)}, NMI: {normalized_mutual_info_score(cell_types, mapped_preds)}, ARI: {adjusted_rand_score(cell_types, mapped_preds)}")