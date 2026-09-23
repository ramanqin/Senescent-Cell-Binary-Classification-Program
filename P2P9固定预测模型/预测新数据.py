"""Use the frozen P2/P9 models on preprocessed, subject-grouped Raman spectra."""
from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from raman_io import read_spectrum


HERE = Path(__file__).resolve().parent
GRID = np.arange(600.0, 1801.0, 1.0)
CLASS_CODE = {"年轻": 0, "衰老": 1}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subject_folders(root: Path) -> list[tuple[str, Path]]:
    grouped = [name for name in CLASS_CODE if (root / name).is_dir()]
    if grouped:
        if any(p.is_dir() and p.name not in CLASS_CODE for p in root.iterdir()):
            raise ValueError("输入目录不能混用有标签和无标签的结构")
        result = [(label, subject) for label in grouped
                  for subject in sorted((root / label).iterdir()) if subject.is_dir()]
    else:
        result = [("未标注", subject) for subject in sorted(root.iterdir()) if subject.is_dir()]
    if not result:
        raise ValueError("没有找到个体文件夹。请检查 README 中的目录结构")
    return result


def load_subjects(root: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    vectors = []
    subjects = []
    for label, folder in subject_folders(root):
        files = sorted(p for p in folder.glob("*.txt") if p.is_file())
        if not files:
            raise ValueError(f"该个体没有TXT光谱：{folder}")
        spectra = []
        for path in files:
            x, intensity = read_spectrum(path)
            if len(x) != len(GRID) or not np.allclose(x, GRID, atol=1e-6, rtol=0):
                raise ValueError(f"波数轴不符，预期600–1800、步长1、共1201点：{path}")
            if not np.isfinite(intensity).all():
                raise ValueError(f"存在非有限强度值：{path}")
            spectra.append(intensity)
        vectors.append(np.mean(np.vstack(spectra), axis=0))
        subjects.append({"编号": folder.name, "原标签": label, "合格光谱数": len(files),
                         "相对路径": folder.relative_to(root).as_posix()})
    return np.vstack(vectors), subjects


def evaluate(rows: list[dict[str, object]]) -> dict[str, object] | None:
    labeled = [r for r in rows if r["原标签"] in CLASS_CODE]
    if not labeled:
        return None
    truth = np.asarray([CLASS_CODE[str(r["原标签"])] for r in labeled])
    predicted = np.asarray([CLASS_CODE[str(r["预测标签"])] for r in labeled])
    scores = np.asarray([float(r["衰老评分"]) for r in labeled])
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return {
        "评价编号数": len(labeled),
        "准确率": float(accuracy_score(truth, predicted)),
        "年轻识别率": float(tn / (tn + fp)) if tn + fp else None,
        "衰老识别率": float(tp / (tp + fn)) if tp + fn else None,
        "平衡准确率": float(((tn / (tn + fp)) + (tp / (tp + fn))) / 2)
        if tn + fp and tp + fn else None,
        "AUC": float(roc_auc_score(truth, scores)) if len(np.unique(truth)) == 2 else None,
        "混淆矩阵_行真实列预测_年轻衰老": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="用固定P2/P9模型预测已预处理的个体平均光谱")
    parser.add_argument("--input", required=True, type=Path, help="有标签或无标签的个体光谱根目录")
    parser.add_argument("--output", type=Path, help="结果目录；默认在输入目录旁边新建")
    args = parser.parse_args()
    source = args.input.resolve()
    if not source.is_dir():
        parser.error(f"输入目录不存在：{source}")
    output = (args.output or source.parent / f"{source.name}_P2P9预测结果").resolve()
    if output == source or output.is_relative_to(source):
        parser.error("结果目录不能放在输入目录内部")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"结果目录已存在且不为空，请换一个目录，避免覆盖原结果：{output}")
    manifest = json.loads((HERE / "模型清单.json").read_text(encoding="utf-8"))
    matrix, subjects = load_subjects(source)
    rows = []
    summary = []
    checks = {}
    with ExitStack() as stack:
        for cls in (Pipeline, PCA, StandardScaler, SVC, LogisticRegression, RandomForestClassifier):
            stack.enter_context(patch.object(cls, "fit", side_effect=AssertionError("预测时禁止拟合模型")))
        for name, spec in manifest["models"].items():
            path = HERE / "模型文件" / spec["file"]
            actual_hash = file_hash(path)
            if actual_hash != spec["sha256"]:
                raise ValueError(f"{name}模型文件SHA-256与清单不符")
            loaded = joblib.load(path)
            model = loaded["model"] if isinstance(loaded, dict) else loaded
            if model.n_features_in_ != 1201 or not np.array_equal(model.classes_, [0, 1]):
                raise ValueError(f"{name}特征维度或类别编码不符")
            predicted = model.predict(matrix)
            scores = (model.decision_function(matrix) if name == "PCA-SVM"
                      else model.predict_proba(matrix)[:, 1])
            model_rows = []
            for subject, guess, score in zip(subjects, predicted, scores):
                row = {"模型": name, **subject, "预测标签": "年轻" if guess == 0 else "衰老",
                       "衰老评分": float(score), "评分类型": spec["score_type"]}
                row["是否正确"] = (row["原标签"] == row["预测标签"]
                                if row["原标签"] in CLASS_CODE else "")
                model_rows.append(row)
            rows.extend(model_rows)
            summary.append({"模型": name, "预测年轻数": int(sum(predicted == 0)),
                            "预测衰老数": int(sum(predicted == 1)), "标签评价": evaluate(model_rows)})
            if file_hash(path) != actual_hash:
                raise AssertionError(f"{name}模型文件在运行期间发生变化")
            checks[name] = {"模型SHA256": actual_hash, "未重新拟合": True}
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "逐编号预测.csv", index=False, encoding="utf-8-sig")
    (output / "预测汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "运行核查.json").write_text(json.dumps({
        "输入目录": str(source), "个体数": len(subjects), "输入矩阵形状": list(matrix.shape),
        "先按个体对合格光谱求平均": True, "使用P6或其他测试集重新训练": False,
        "PCA或标准化重新拟合": False, "阈值调整": False, "模型核查": checks,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{len(subjects)}个编号，结果在 {output}")
    for item in summary:
        print(f"{item['模型']}：年轻 {item['预测年轻数']}，衰老 {item['预测衰老数']}")


if __name__ == "__main__":
    main()
