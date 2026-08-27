"""基于已完成人工QC标注的轻量决策树辅助器。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


TREE_FEATURE_FIELDS = [
    "n_points",
    "x_min",
    "x_max",
    "x_step_median",
    "intensity_std",
    "intensity_p05",
    "intensity_median",
    "intensity_p95",
    "robust_intensity_range",
    "robust_snr_db",
    "silent_noise_mad",
    "fingerprint_contrast",
    "ch_contrast",
    "spike_count_auto",
    "saturation_run_max",
    "axis_monotonic",
    "missing_count",
    "finite_fraction",
]

MIN_TRAINING_SAMPLES = 30
MIN_SAMPLES_PER_CLASS = 8
HIGH_CONFIDENCE = 0.90


@dataclass
class QCTreeModel:
    classifier: DecisionTreeClassifier
    medians: pd.Series
    sample_count: int
    pass_count: int
    attention_count: int


def _numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    for field in TREE_FEATURE_FIELDS:
        source = frame[field] if field in frame.columns else pd.Series(np.nan, index=frame.index)
        features[field] = pd.to_numeric(source, errors="coerce")
    return features.replace([np.inf, -np.inf], np.nan)


def train_qc_tree(annotations: pd.DataFrame | None) -> tuple[QCTreeModel | None, str]:
    """训练PASS与“需关注(REVIEW/FAIL)”二分类树；数据不足时不启用。"""
    if annotations is None or annotations.empty or "qc_status" not in annotations:
        return None, f"决策树待训练：至少需要{MIN_TRAINING_SAMPLES}条人工标注"

    status = annotations["qc_status"].astype(str).str.lower()
    completed = status.isin({"pass", "review", "fail"})
    training = annotations.loc[completed].copy()
    status = status.loc[completed]
    pass_count = int(status.eq("pass").sum())
    attention_count = int(status.isin({"review", "fail"}).sum())
    total = len(training)

    if total < MIN_TRAINING_SAMPLES or min(pass_count, attention_count) < MIN_SAMPLES_PER_CLASS:
        return (
            None,
            f"决策树待训练：当前{total}条（PASS {pass_count}，需关注 {attention_count}）；"
            f"要求总数≥{MIN_TRAINING_SAMPLES}且两类各≥{MIN_SAMPLES_PER_CLASS}",
        )

    features = _numeric_features(training)
    medians = features.median(axis=0, skipna=True).fillna(0.0)
    features = features.fillna(medians)
    target = status.eq("pass").astype(int)

    classifier = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=max(3, min(10, total // 12)),
        class_weight="balanced",
        random_state=20260814,
    )
    classifier.fit(features[TREE_FEATURE_FIELDS], target)
    return (
        QCTreeModel(classifier, medians, total, pass_count, attention_count),
        f"决策树已训练：{total}条（PASS {pass_count}，需关注 {attention_count}）",
    )


def predict_qc(model: QCTreeModel, metrics: dict) -> dict:
    """返回辅助建议；预测结果不写入标注变量或CSV。"""
    features = _numeric_features(pd.DataFrame([metrics])).fillna(model.medians)
    probabilities = model.classifier.predict_proba(features[TREE_FEATURE_FIELDS])[0]
    probability_by_class = {
        int(label): float(probability)
        for label, probability in zip(model.classifier.classes_, probabilities)
    }
    pass_probability = probability_by_class.get(1, 0.0)
    attention_probability = probability_by_class.get(0, 0.0)

    if pass_probability >= HIGH_CONFIDENCE:
        suggestion = "pass"
        text = f"决策树建议PASS（{pass_probability:.0%}）；请看图确认后采用"
    elif attention_probability >= HIGH_CONFIDENCE:
        suggestion = "attention"
        text = f"决策树建议人工关注（{attention_probability:.0%}）；请判断原因及REVIEW/FAIL"
    else:
        suggestion = "uncertain"
        text = f"决策树不确定（PASS概率 {pass_probability:.0%}）；请人工判断"

    return {
        "suggestion": suggestion,
        "text": text,
        "pass_probability": pass_probability,
        "attention_probability": attention_probability,
    }
