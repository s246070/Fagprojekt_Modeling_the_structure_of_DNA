"""Run KMeans clustering on a saved model embedding and report metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "peakvi_latent_2d_full.pth"
DEFAULT_LABELS_PATH = ROOT_DIR / "src" / "benchmarking" / "cell_types.txt"


def load_embeddings(model_path: Path) -> np.ndarray:
	"""Load a 2D embedding matrix from a saved model file.

	The repository stores either raw latent arrays or checkpoint dictionaries.
	This helper accepts both formats and normalizes the output to a 2D NumPy
	array suitable for clustering.
	"""

	loaded = torch.load(model_path, map_location="cpu")

	if isinstance(loaded, dict):
		for key in ("embed_cells", "latent", "latent_representation", "X_peakvi", "embedding", "embeddings"):
			if key in loaded:
				loaded = loaded[key]
				break
		else:
			raise ValueError(
				f"Could not find an embedding tensor in {model_path.name}. Available keys: {list(loaded.keys())}"
			)

	if hasattr(loaded, "detach"):
		array = loaded.detach().cpu().numpy()
	else:
		array = np.asarray(loaded)

	if array.ndim == 1:
		array = array[:, None]
	elif array.ndim > 2:
		array = array.reshape(array.shape[0], -1)

	return array


def load_labels(labels_path: Path) -> list[str]:
	"""Load the ground-truth labels used by the existing benchmarking scripts."""

	with labels_path.open("r", encoding="utf-8") as handle:
		return [line.strip() for line in handle]


def clustering_accuracy(true_labels: list[str], cluster_labels: np.ndarray) -> float:
	"""Compute best-match clustering accuracy using the Hungarian algorithm."""

	label_to_index = {label: index for index, label in enumerate(sorted(set(true_labels)))}
	true_indices = np.array([label_to_index[label] for label in true_labels])

	matrix = contingency_matrix(true_indices, cluster_labels)
	row_ind, col_ind = linear_sum_assignment(-matrix)
	matched = matrix[row_ind, col_ind].sum()
	return float(matched / len(true_labels))


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--model-path",
		type=Path,
		default=DEFAULT_MODEL_PATH,
		help="Path to a saved model or latent representation in the models folder.",
	)
	parser.add_argument(
		"--labels-path",
		type=Path,
		default=DEFAULT_LABELS_PATH,
		help="Path to the matching cell-type labels file.",
	)
	parser.add_argument(
		"--n-clusters",
		type=int,
		default=None,
		help="Number of clusters to fit. Defaults to the number of unique labels.",
	)
	return parser.parse_args()


def main() -> None:
	"""Run KMeans on a saved model embedding and print evaluation metrics."""

	args = parse_args()
	embeddings = load_embeddings(args.model_path)
	labels = load_labels(args.labels_path)

	if embeddings.shape[0] != len(labels):
		raise ValueError(
			f"Embedding rows ({embeddings.shape[0]}) do not match labels ({len(labels)})."
		)

	n_clusters = args.n_clusters or len(set(labels))
	kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
	cluster_labels = kmeans.fit_predict(embeddings)

	print(f"Model: {args.model_path}")
	print(f"Samples: {embeddings.shape[0]}, features: {embeddings.shape[1]}, clusters: {n_clusters}")
	print(f"Accuracy: {clustering_accuracy(labels, cluster_labels):.4f}")
	print(f"NMI: {normalized_mutual_info_score(labels, cluster_labels):.4f}")
	print(f"ARI: {adjusted_rand_score(labels, cluster_labels):.4f}")


if __name__ == "__main__":
	main()
