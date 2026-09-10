from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from raman_io import SpectralDataset, load_subject_dataset


MODEL_ALIASES = {
    "svm": "pca_svm",
    "logistic": "pca_logistic_regression",
    "random_forest": "random_forest",
}


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def model_specification(model_name: str, pca_variance: float, seed: int):
    if model_name == "svm":
        estimator = Pipeline(
            [
                ("pca", PCA(n_components=pca_variance, svd_solver="full")),
                ("scale", StandardScaler()),
                ("model", SVC(class_weight="balanced")),
            ]
        )
        grid = [
            {
                "model__kernel": ["linear"],
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
            {
                "model__kernel": ["rbf"],
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__gamma": ["scale", 0.01, 0.1, 1.0],
            },
        ]
        return estimator, grid

    if model_name == "logistic":
        estimator = Pipeline(
            [
                ("pca", PCA(n_components=pca_variance, svd_solver="full")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        class_weight="balanced",
                        max_iter=20000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        grid = {
            "model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        }
        return estimator, grid

    if model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=500,
            criterion="gini",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
        grid = {
            "max_depth": [None, 4, 8],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.25],
        }
        return estimator, grid

    raise ValueError(f"Unknown model: {model_name}")


def score_estimator(estimator, matrix: np.ndarray, model_name: str) -> np.ndarray:
    if model_name == "svm":
        return estimator.decision_function(matrix)
    return estimator.predict_proba(matrix)[:, 1]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity_positive": float(recall_score(y_true, y_pred, pos_label=1)),
        "specificity_negative": float(tn / (tn + fp)),
        "precision_positive": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_positive": float(f1_score(y_true, y_pred, pos_label=1)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def save_figures(
    output_dir: Path,
    model_title: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    class_names: tuple[str, str],
) -> None:
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_value = roc_auc_score(labels, scores)
    fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=180)
    ax.plot(fpr, tpr, linewidth=2.5, label=f"{model_title} (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{model_title}: outer-fold ROC")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc.png", bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    normalized = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.8), dpi=180)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    # Generic plot labels avoid missing-glyph warnings on systems without CJK fonts.
    plot_class_names = ["Negative (0)", "Positive (1)"]
    ax.set_xticks([0, 1], labels=plot_class_names)
    ax.set_yticks([0, 1], labels=plot_class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{model_title}: outer-fold confusion matrix")
    for row in range(2):
        for column in range(2):
            color = "white" if normalized[row, column] > 0.5 else "black"
            ax.text(
                column,
                row,
                f"{cm[row, column]}\n{normalized[row, column]:.1%}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=color,
            )
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)


def nested_cross_validation(
    dataset: SpectralDataset,
    model_name: str,
    output_root: Path,
    outer_splits: int,
    inner_splits: int,
    seed: int,
    pca_variance: float,
) -> dict:
    matrix, labels = dataset.matrix, dataset.labels
    estimator, grid = model_specification(model_name, pca_variance, seed)
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    predictions = np.full(len(labels), -1, dtype=int)
    scores = np.full(len(labels), np.nan, dtype=float)
    fold_ids = np.full(len(labels), -1, dtype=int)
    fold_results = []

    for fold, (train_index, test_index) in enumerate(outer_cv.split(matrix, labels), start=1):
        inner_cv = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=seed + fold,
        )
        search = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            scoring="balanced_accuracy",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(matrix[train_index], labels[train_index])
        fold_predictions = search.predict(matrix[test_index])
        fold_scores = score_estimator(search.best_estimator_, matrix[test_index], model_name)
        predictions[test_index] = fold_predictions
        scores[test_index] = fold_scores
        fold_ids[test_index] = fold
        fold_metrics = calculate_metrics(labels[test_index], fold_predictions, fold_scores)
        fold_results.append(
            {
                "fold": fold,
                "train_subjects": int(len(train_index)),
                "test_subjects": int(len(test_index)),
                "inner_best_balanced_accuracy": float(search.best_score_),
                "best_parameters": json_ready(search.best_params_),
                **fold_metrics,
            }
        )

    if (predictions < 0).any() or np.isnan(scores).any():
        raise RuntimeError("Outer-fold predictions are incomplete")

    model_output = output_root / MODEL_ALIASES[model_name]
    model_output.mkdir(parents=True, exist_ok=True)
    metrics = calculate_metrics(labels, predictions, scores)
    pd.DataFrame(fold_results).to_json(
        model_output / "fold_results.json", orient="records", indent=2
    )
    pd.DataFrame(
        {
            "sample_id": dataset.sample_ids,
            "spectra_count": dataset.spectra_counts,
            "outer_fold": fold_ids,
            "true_class_code": labels,
            "true_class": [dataset.class_names[value] for value in labels],
            "score_positive": scores,
            "predicted_class_code": predictions,
            "predicted_class": [dataset.class_names[value] for value in predictions],
            "correct": predictions == labels,
        }
    ).to_csv(model_output / "predictions.csv", index=False, encoding="utf-8-sig")

    final_inner = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=seed + 100,
    )
    final_search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring="balanced_accuracy",
        cv=final_inner,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    final_search.fit(matrix, labels)
    joblib.dump(final_search.best_estimator_, model_output / "model.joblib")

    result = {
        "model": MODEL_ALIASES[model_name],
        "independent_subjects": int(len(labels)),
        "class_names": dataset.class_names,
        "class_counts": np.bincount(labels, minlength=2).tolist(),
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "random_seed": seed,
        "metrics": metrics,
        "final_best_parameters": json_ready(final_search.best_params_),
        "final_inner_score": float(final_search.best_score_),
    }
    (model_output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_figures(
        model_output,
        MODEL_ALIASES[model_name],
        labels,
        predictions,
        scores,
        dataset.class_names,
    )
    return result


def parse_arguments():
    parser = argparse.ArgumentParser(description="Subject-level Raman binary classification")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_ALIASES),
        default=None,
    )
    parser.add_argument("--negative-class")
    parser.add_argument("--positive-class")
    parser.add_argument("--outer-splits", type=int)
    parser.add_argument("--inner-splits", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--pca-variance", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = {}
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    input_dir = args.input or config.get("input_dir")
    output_dir = args.output or config.get("output_dir", "results/new_run")
    if input_dir is None:
        raise ValueError("Provide --input or input_dir in --config")
    negative_class = args.negative_class or config.get("negative_class", "年轻")
    positive_class = args.positive_class or config.get("positive_class", "衰老")
    outer_splits = args.outer_splits or int(config.get("outer_splits", 5))
    inner_splits = args.inner_splits or int(config.get("inner_splits", 4))
    seed = args.seed if args.seed is not None else int(config.get("random_seed", 42))
    pca_variance = args.pca_variance or float(config.get("pca_variance", 0.95))
    models = args.models or ["svm", "logistic", "random_forest"]

    dataset = load_subject_dataset(input_dir, negative_class, positive_class)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for model_name in models:
        print(f"Running {model_name}...")
        results.append(
            nested_cross_validation(
                dataset,
                model_name,
                output_root,
                outer_splits,
                inner_splits,
                seed,
                pca_variance,
            )
        )
    summary_rows = [
        {"model": result["model"], **result["metrics"]} for result in results
    ]
    pd.DataFrame(summary_rows).to_csv(
        output_root / "model_summary.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
