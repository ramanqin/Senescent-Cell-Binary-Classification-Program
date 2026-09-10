from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from raman_io import load_subject_dataset


COLORS = ("#2474B5", "#D95F45")
MARKERS = ("o", "^")


def scatter(ax, coordinates, labels, class_names):
    for code in (0, 1):
        mask = labels == code
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=52,
            color=COLORS[code],
            marker=MARKERS[code],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
            label=f"Class {code} ({'negative' if code == 0 else 'positive'}, n={mask.sum()})",
        )
    ax.grid(alpha=0.18)
    ax.legend(loc="best")


def classical_mds(distance: np.ndarray):
    n = len(distance)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distance ** 2) @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    positive = values > 1e-12
    values, vectors = values[positive], vectors[:, positive]
    coordinates = vectors[:, :2] * np.sqrt(values[:2])
    shares = values[:2] / values.sum()
    return coordinates, shares


def parse_arguments():
    parser = argparse.ArgumentParser(description="PCA, t-SNE and RF-proximity visualization")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--negative-class", default="年轻")
    parser.add_argument("--positive-class", default="衰老")
    parser.add_argument("--perplexity", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-model", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_subject_dataset(args.input, args.negative_class, args.positive_class)
    matrix, labels = data.matrix, data.labels
    summary = {}

    pca = PCA(n_components=2, svd_solver="full")
    pca_coordinates = pca.fit_transform(matrix)
    fig, ax = plt.subplots(figsize=(7.4, 6.2), dpi=180)
    scatter(ax, pca_coordinates, labels, data.class_names)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Unsupervised PCA of subject-average spectra")
    fig.tight_layout()
    fig.savefig(args.output / "pca_2d.png", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(
        {
            "sample_id": data.sample_ids,
            "class_code": labels,
            "PC1": pca_coordinates[:, 0],
            "PC2": pca_coordinates[:, 1],
        }
    ).to_csv(args.output / "pca_coordinates.csv", index=False, encoding="utf-8-sig")
    summary["pca"] = {
        "supervised": False,
        "PC1": float(pca.explained_variance_ratio_[0]),
        "PC2": float(pca.explained_variance_ratio_[1]),
    }

    pca95 = PCA(n_components=0.95, svd_solver="full")
    features = StandardScaler().fit_transform(pca95.fit_transform(matrix))
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        early_exaggeration=12.0,
        learning_rate="auto",
        max_iter=1500,
        init="pca",
        random_state=args.seed,
    )
    tsne_coordinates = tsne.fit_transform(features)
    tsne_silhouette = silhouette_score(tsne_coordinates, labels)
    fig, ax = plt.subplots(figsize=(7.4, 6.2), dpi=180)
    scatter(ax, tsne_coordinates, labels, data.class_names)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title(f"Exploratory t-SNE (perplexity={args.perplexity:g})")
    fig.tight_layout()
    fig.savefig(args.output / "tsne_2d.png", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(
        {
            "sample_id": data.sample_ids,
            "class_code": labels,
            "tsne_1": tsne_coordinates[:, 0],
            "tsne_2": tsne_coordinates[:, 1],
        }
    ).to_csv(args.output / "tsne_coordinates.csv", index=False, encoding="utf-8-sig")
    summary["tsne"] = {
        "supervised": False,
        "global_embedding_warning": True,
        "pca95_components": int(pca95.n_components_),
        "perplexity": args.perplexity,
        "random_seed": args.seed,
        "silhouette_by_class_label": float(tsne_silhouette),
    }

    if args.rf_model:
        forest = joblib.load(args.rf_model)
        leaf_ids = forest.apply(matrix)
        proximity = np.zeros((len(labels), len(labels)), dtype=float)
        for tree_index in range(leaf_ids.shape[1]):
            leaves = leaf_ids[:, tree_index]
            proximity += leaves[:, None] == leaves[None, :]
        proximity /= leaf_ids.shape[1]
        distance = 1.0 - proximity
        rf_coordinates, rf_shares = classical_mds(distance)
        rf_silhouette = silhouette_score(rf_coordinates, labels)

        fig, ax = plt.subplots(figsize=(7.4, 6.2), dpi=180)
        scatter(ax, rf_coordinates, labels, data.class_names)
        ax.set_xlabel(f"PCo1 ({rf_shares[0]:.1%})")
        ax.set_ylabel(f"PCo2 ({rf_shares[1]:.1%})")
        ax.set_title("Random-forest proximity PCoA/MDS")
        fig.tight_layout()
        fig.savefig(args.output / "rf_proximity_pcoa.png", bbox_inches="tight")
        plt.close(fig)

        order = np.argsort(labels, kind="stable")
        ordered = proximity[np.ix_(order, order)]
        fig, ax = plt.subplots(figsize=(7.0, 6.3), dpi=180)
        image = ax.imshow(ordered, cmap="YlGnBu", vmin=0, vmax=1)
        boundary = int((labels == 0).sum()) - 0.5
        ax.axhline(boundary, color="white", linewidth=1.4)
        ax.axvline(boundary, color="white", linewidth=1.4)
        fig.colorbar(image, ax=ax, label="Same-leaf proportion")
        ax.set_title("Random-forest proximity matrix")
        ax.set_xlabel("Subjects sorted by true class")
        ax.set_ylabel("Subjects sorted by true class")
        fig.tight_layout()
        fig.savefig(args.output / "rf_proximity_heatmap.png", bbox_inches="tight")
        plt.close(fig)

        pd.DataFrame(
            {
                "sample_id": data.sample_ids,
                "class_code": labels,
                "PCo1": rf_coordinates[:, 0],
                "PCo2": rf_coordinates[:, 1],
            }
        ).to_csv(
            args.output / "rf_proximity_coordinates.csv", index=False, encoding="utf-8-sig"
        )
        summary["random_forest_proximity"] = {
            "supervised": True,
            "full_data_fit_warning": True,
            "trees": int(leaf_ids.shape[1]),
            "PCo1": float(rf_shares[0]),
            "PCo2": float(rf_shares[1]),
            "silhouette_by_class_label": float(rf_silhouette),
        }

    (args.output / "embedding_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
