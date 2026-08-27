from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    for key in ("input_dir", "output_dir"):
        value = Path(config[key])
        if not value.is_absolute():
            value = (config_path.parent / value).resolve()
        config[key] = str(value)

    defaults = {
        "extensions": ["txt", "csv", "dat"],
        "sample_depth_after_class": 1,
        "outer_splits": 5,
        "inner_splits": 4,
        "random_seed": 42,
        "pca_variance": 0.95,
        "class_weight": "balanced",
        "n_jobs": -1,
        "c_values": [0.01, 0.1, 1.0, 10.0, 100.0],
        "gamma_values": ["scale", 0.01, 0.1, 1.0],
        "roc_filename": "ROC.png",
        "grid_tolerance": 1e-6,
        "min_points": 20,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)

    required = ["input_dir", "output_dir", "negative_class", "positive_class"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"配置缺少字段：{missing}")
    if not Path(config["input_dir"]).is_dir():
        raise FileNotFoundError(f"输入文件夹不存在：{config['input_dir']}")
    if not 0 < float(config["pca_variance"]) < 1:
        raise ValueError("pca_variance必须在0与1之间，例如0.95")
    return config


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法读取文件：{path}")


def read_spectrum(path: Path, min_points: int) -> tuple[np.ndarray, np.ndarray]:
    """读取前两列数值；表头和非数值行自动跳过。"""
    rows: list[tuple[float, float]] = []
    for line in read_text(path).splitlines():
        if FLOAT_PATTERN.match(line.lstrip()) is None:
            continue
        values = FLOAT_PATTERN.findall(line)
        if len(values) >= 2:
            rows.append((float(values[0]), float(values[1])))
    if len(rows) < min_points:
        raise ValueError(f"有效数据点不足：{path}")

    data = np.asarray(rows, dtype=float)
    data = data[np.isfinite(data).all(axis=1)]
    data = data[np.argsort(data[:, 0])]
    x, y = data[:, 0], data[:, 1]

    unique_x, inverse = np.unique(x, return_inverse=True)
    if len(unique_x) != len(x):
        y = np.bincount(inverse, weights=y) / np.bincount(inverse)
        x = unique_x
    if len(x) < min_points or np.any(np.diff(x) <= 0):
        raise ValueError(f"波数列无效：{path}")
    return x, y


def class_lookup(config: dict) -> dict[str, tuple[int, str]]:
    lookup: dict[str, tuple[int, str]] = {}
    for code, key in [(0, "negative_class"), (1, "positive_class")]:
        item = config[key]
        name = item["name"]
        for alias in set(item["folders"] + [name]):
            normalized = alias.casefold()
            if normalized in lookup and lookup[normalized][0] != code:
                raise ValueError(f"两个类别使用了相同目录别名：{alias}")
            lookup[normalized] = (code, name)
    return lookup


def identify_sample(path: Path, input_root: Path, config: dict) -> tuple[int, str, str, str, str]:
    parts = path.relative_to(input_root).parts
    lookup = class_lookup(config)
    depth = int(config["sample_depth_after_class"])
    for index, part in enumerate(parts[:-1]):
        matched = lookup.get(part.casefold())
        if matched is None:
            continue
        code, class_name = matched
        sample_index = index + depth
        if sample_index < len(parts) - 1:
            folder = parts[sample_index]
        elif index + 1 == len(parts) - 1:
            folder = path.stem
        else:
            raise ValueError(f"类别目录后没有找到样本文件夹：{path}")
        prefix = parts[:index]
        batch_id = "/".join(prefix) if prefix else input_root.name
        sample_id = f"{batch_id}/{class_name}/{folder}"
        return code, class_name, folder, batch_id, sample_id
    raise ValueError(f"路径中没有找到配置的类别目录：{path}")


def load_samples(config: dict) -> dict:
    input_root = Path(config["input_dir"])
    output_root = Path(config["output_dir"]).resolve()
    extensions = {"." + value.lower().lstrip(".") for value in config["extensions"]}
    excluded_names = {"run_manifest.csv", "preprocessing_manifest.csv", "predictions.csv", "metrics.json"}
    files = sorted(
        path for path in input_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
        and not path.resolve().is_relative_to(output_root)
        and path.name.casefold() not in excluded_names
        and not path.name.startswith("处理报告_")
        and not any(
            part.casefold() in {"failed", "result_plot", "__pycache__"}
            for part in path.relative_to(input_root).parts[:-1]
        )
    )
    if not files:
        raise FileNotFoundError("输入目录中没有找到光谱文件")

    records: list[tuple[str, Path, np.ndarray, np.ndarray]] = []
    metadata: dict[str, tuple[int, str, str, str]] = {}
    spectrum_manifest: list[dict] = []

    for path in files:
        code, class_name, folder, batch_id, sample_id = identify_sample(path, input_root, config)
        x, y = read_spectrum(path, int(config["min_points"]))
        records.append((sample_id, path, x, y))
        metadata[sample_id] = (code, class_name, folder, batch_id)
        spectrum_manifest.append({
            "stage": "modeling_input",
            "source_root": str(input_root.resolve()),
            "source_path": str(path.resolve()),
            "relative_path": path.relative_to(input_root).as_posix(),
            "batch_id": batch_id,
            "sample_id": sample_id,
            "sample_folder": folder,
            "class_code": code,
            "class_name": class_name,
            "status": "included",
        })

    first_x = records[0][2]
    tolerance = float(config["grid_tolerance"])
    grids_are_equal = all(
        len(x) == len(first_x) and np.allclose(x, first_x, rtol=0, atol=tolerance)
        for _, _, x, _ in records[1:]
    )
    grid_signatures = {
        (len(x), round(float(x[0]), 6), round(float(x[-1]), 6),
         round(float(np.median(np.diff(x))), 6))
        for _, _, x, _ in records
    }

    if not grids_are_equal:
        _, mismatch_path, mismatch_x, _ = next(
            record for record in records[1:]
            if len(record[2]) != len(first_x)
            or not np.allclose(record[2], first_x, rtol=0, atol=tolerance)
        )
        reference_path = records[0][1]
        raise ValueError(
            "波数网格不一致，PCA-SVM不会自动插值。参考光谱："
            f"{reference_path}，{len(first_x)}点，"
            f"{first_x[0]:.3f}–{first_x[-1]:.3f} cm⁻¹；"
            f"不一致文件：{mismatch_path}，{len(mismatch_x)}点，"
            f"{mismatch_x[0]:.3f}–{mismatch_x[-1]:.3f} cm⁻¹。"
            "请先在光谱预处理软件中对全部光谱采用同一范围和步长重采样。"
        )

    reference_x = first_x
    aligned_records = [(sample_id, y) for sample_id, _, _, y in records]

    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for sample_id, y in aligned_records:
        grouped[sample_id].append(y)

    sample_ids = sorted(grouped, key=lambda key: (metadata[key][0], key))
    matrix = np.vstack([np.mean(grouped[key], axis=0) for key in sample_ids])
    labels = np.asarray([metadata[key][0] for key in sample_ids], dtype=int)
    class_names = [metadata[key][1] for key in sample_ids]
    folders = [metadata[key][2] for key in sample_ids]
    batch_ids = [metadata[key][3] for key in sample_ids]
    spectra_counts = [len(grouped[key]) for key in sample_ids]

    counts = np.bincount(labels, minlength=2)
    if counts.min() < 2:
        raise ValueError("每个类别至少需要2个独立样本文件夹")
    return {
        "x": reference_x,
        "matrix": matrix,
        "labels": labels,
        "sample_ids": sample_ids,
        "class_names": class_names,
        "folders": folders,
        "batch_ids": batch_ids,
        "spectra_counts": spectra_counts,
        "spectrum_manifest": spectrum_manifest,
        "grid_info": {
            "original_grid_count": len(grid_signatures),
            "validated_equal": True,
            "points": int(len(reference_x)),
            "start": float(reference_x[0]),
            "end": float(reference_x[-1]),
            "median_step": float(np.median(np.diff(reference_x))),
        },
    }


def make_search(config: dict, inner_cv: StratifiedKFold) -> GridSearchCV:
    pipeline = Pipeline([
        ("pca", PCA(n_components=float(config["pca_variance"]), svd_solver="full")),
        ("scale", StandardScaler()),
        ("svm", SVC(class_weight=config["class_weight"], cache_size=500)),
    ])
    grid = [
        {"svm__kernel": ["linear"], "svm__C": config["c_values"]},
        {
            "svm__kernel": ["rbf"],
            "svm__C": config["c_values"],
            "svm__gamma": config["gamma_values"],
        },
    ]
    return GridSearchCV(
        pipeline,
        param_grid=grid,
        scoring="balanced_accuracy",
        cv=inner_cv,
        n_jobs=int(config["n_jobs"]),
        refit=True,
        error_score="raise",
    )


def nested_cross_validation(data: dict, config: dict) -> tuple[np.ndarray, list[dict]]:
    x, y = data["matrix"], data["labels"]
    minority = int(np.bincount(y, minlength=2).min())
    outer_splits = min(int(config["outer_splits"]), minority)
    if outer_splits < 2:
        raise ValueError("独立样本不足，无法进行外层交叉验证")
    outer = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=int(config["random_seed"]),
    )
    scores = np.full(len(y), np.nan)
    fold_parameters: list[dict] = []

    for fold, (train_index, test_index) in enumerate(outer.split(x, y), start=1):
        train_y = y[train_index]
        inner_splits = min(int(config["inner_splits"]), int(np.bincount(train_y).min()))
        if inner_splits < 2:
            raise ValueError("外层训练集的少数类别不足，不能进行内层调参")
        inner = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=int(config["random_seed"]) + fold * 101,
        )
        search = make_search(config, inner)
        search.fit(x[train_index], train_y)
        scores[test_index] = search.best_estimator_.decision_function(x[test_index])
        fold_parameters.append({
            "fold": fold,
            "inner_best_balanced_accuracy": float(search.best_score_),
            "best_parameters": search.best_params_,
        })
    if np.isnan(scores).any():
        raise RuntimeError("外层交叉验证未覆盖全部样本")
    return scores, fold_parameters


def save_roc(y: np.ndarray, scores: np.ndarray, config: dict, output_dir: Path) -> dict:
    predicted = (scores >= 0).astype(int)
    auc_value = float(roc_auc_score(y, scores))
    balanced = float(balanced_accuracy_score(y, predicted))
    fpr, tpr, _ = roc_curve(y, scores)

    negative_name = config["negative_class"]["name"]
    positive_name = config["positive_class"]["name"]
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.plot(fpr, tpr, color="#C44E52", linewidth=2.2, label=f"PCA-SVM，AUC={auc_value:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="0.55", label="随机水平")
    ax.set_xlabel("假阳性率")
    ax.set_ylabel("真阳性率")
    ax.set_title(f"{negative_name} vs {positive_name} ROC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    ax.legend()
    roc_path = output_dir / config["roc_filename"]
    fig.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": auc_value,
        "balanced_accuracy": balanced,
        "accuracy": float(accuracy_score(y, predicted)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "roc_file": str(roc_path),
    }


def save_predictions(data: dict, scores: np.ndarray, config: dict, output_dir: Path) -> None:
    predicted = (scores >= 0).astype(int)
    names = {0: config["negative_class"]["name"], 1: config["positive_class"]["name"]}
    path = output_dir / "predictions.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "sample_id", "batch_id", "folder", "spectra_count", "true_class",
            "decision_score_positive", "predicted_class", "correct",
        ])
        writer.writeheader()
        for sample_id, batch_id, folder, count, true_code, score, pred_code in zip(
            data["sample_ids"], data["batch_ids"], data["folders"], data["spectra_counts"],
            data["labels"], scores, predicted,
        ):
            writer.writerow({
                "sample_id": sample_id,
                "batch_id": batch_id,
                "folder": folder,
                "spectra_count": count,
                "true_class": names[int(true_code)],
                "decision_score_positive": float(score),
                "predicted_class": names[int(pred_code)],
                "correct": bool(true_code == pred_code),
            })


def create_analysis_run(output_root: Path) -> tuple[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"analysis_run_{timestamp}"
    output_dir = output_root / run_id
    suffix = 1
    while output_dir.exists():
        run_id = f"analysis_run_{timestamp}_{suffix:02d}"
        output_dir = output_root / run_id
        suffix += 1
    output_dir.mkdir(parents=True)
    return run_id, output_dir


def save_run_manifest(data: dict, run_id: str, output_dir: Path) -> None:
    fields = [
        "run_id", "upstream_run_id", "stage", "source_root", "source_path",
        "relative_path", "batch_id", "sample_id", "sample_folder",
        "class_code", "class_name", "status",
    ]
    upstream_run_id = Path(data["spectrum_manifest"][0]["source_root"]).name if data["spectrum_manifest"] else ""
    with (output_dir / "run_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in data["spectrum_manifest"]:
            writer.writerow({
                "run_id": run_id,
                "upstream_run_id": upstream_run_id,
                **row,
            })


def fit_final_model(data: dict, config: dict, output_dir: Path) -> dict:
    y = data["labels"]
    inner_splits = min(int(config["inner_splits"]), int(np.bincount(y).min()))
    inner = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=int(config["random_seed"]) + 2026,
    )
    search = make_search(config, inner)
    search.fit(data["matrix"], y)
    joblib.dump({
        "model": search.best_estimator_,
        "wave_numbers": data["x"],
        "negative_class": config["negative_class"]["name"],
        "positive_class": config["positive_class"]["name"],
        "best_parameters": search.best_params_,
    }, output_dir / "model.joblib")
    return {
        "best_parameters": search.best_params_,
        "inner_selection_score": float(search.best_score_),
    }


def run_analysis(config: dict) -> dict:
    """运行完整分析，供命令行和简易前端共同调用。"""
    data = load_samples(config)
    scores, fold_parameters = nested_cross_validation(data, config)
    # 只有数据读取和交叉验证成功后才创建结果批次，避免失败运行留下空目录。
    run_id, output_dir = create_analysis_run(Path(config["output_dir"]))
    metrics = save_roc(data["labels"], scores, config, output_dir)
    save_predictions(data, scores, config, output_dir)
    save_run_manifest(data, run_id, output_dir)
    final_model = fit_final_model(data, config, output_dir)

    result = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "independent_samples": int(len(data["labels"])),
        "spectra_total": int(sum(data["spectra_counts"])),
        "wave_number_grid": data["grid_info"],
        "metrics": metrics,
        "outer_fold_parameters": fold_parameters,
        "final_model": final_model,
        "note": "最终模型的内层选择成绩不是外层测试成绩。",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "run_parameters.json").write_text(
        json.dumps({"run_id": run_id, "stage": "modeling", "config": config}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("用法：python pca_svm_analysis.py config.json")
    config = load_config(sys.argv[1])
    result = run_analysis(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nROC图已保存：{result['metrics']['roc_file']}")


if __name__ == "__main__":
    main()
