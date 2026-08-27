from __future__ import annotations


import hashlib
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


TOOL_VERSION = "2.3.0-qc-assistant"

# 全项目统一：年轻为阴性类0，衰老为阳性类1。
CLASS_MAP = {"young": 0, "aging": 1}
CLASS_ALIASES = {"aging": "aging", "young": "young", "0": "young", "1": "aging"}
QC_STATUSES = {"pass", "review", "fail"}

ISSUE_FIELDS = [
    "vague_finger_peak",
    "missing_finger_peak",
    "low_finger_peak",
    "low_snr",
    "missing_ch_peak",
    "low_ch_peak",
    "cosmic_rays",
]

METRIC_FIELDS = [
    "n_points",
    "x_min",
    "x_max",
    "x_step_median",
    "intensity_min",
    "intensity_max",
    "intensity_mean",
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

SAMPLE_MANIFEST_COLUMNS = [
    "spectrum_uid",
    "batch_id",
    "class_original",
    "class_binary",
    "subject_id",
    "spectrum_no",
    "file_name",
    "file_relative_path",
    "file_absolute_path",
    "selected_folder",
    "tif_path",
    "png_path",
    "sampling_strategy",
    "sampling_seed",
    "sample_order",
]

OUTPUT_COLUMNS = [
    "spectrum_uid",
    "batch_id",
    "class_original",
    "class_binary",
    "subject_id",
    "spectrum_no",
    "file_name",
    "file_relative_path",
    "file_absolute_path",
    "selected_folder",
    "tif_path",
    "png_path",
    "sampling_strategy",
    "sampling_seed",
    "sample_order",
    "qc_status",
    *ISSUE_FIELDS,
    "other_mark",
    *METRIC_FIELDS,
    "annotated_at",
    "tool_version",
]


def _natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value)]


def _spectrum_number(stem: str) -> int | None:
    match = re.search(r"-(\d+)_spec$", stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _safe_uid(batch: str, group: str, subject: str, spectrum_no: int | None, stem: str) -> str:
    batch_safe = re.sub(r"[^A-Za-z0-9]+", "_", batch).strip("_") or "batch"
    group_code = {"young": "Y", "aging": "A"}.get(group, "U")
    subject_safe = re.sub(r"[^A-Za-z0-9]+", "_", subject).strip("_") or "unknown"
    suffix = f"S{spectrum_no:03d}" if spectrum_no is not None else hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
    return f"{batch_safe}_{group_code}_{subject_safe}_{suffix}"


def resolve_input_folder(selected_path: str | Path) -> Path:
    """校验并规范化用户选择的任意TXT递归导入文件夹。"""
    selected = Path(selected_path).expanduser().resolve()
    if not selected.is_dir():
        raise ValueError(f"所选路径不存在或不是文件夹：{selected}")
    return selected


resolve_data_root = resolve_input_folder


def _infer_metadata(txt_path: Path, selected_root: Path) -> tuple[str, int | str, str, str, Path | None]:
    """尽可能从TXT路径中解析组别、标签、个体和批次。"""
    parts = list(txt_path.resolve().parts)
    folded = [part.casefold() for part in parts]
    class_index = next((index for index in range(len(parts) - 1, -1, -1) if folded[index] in CLASS_ALIASES), None)
    if class_index is not None:
        group = CLASS_ALIASES[folded[class_index]]
        class_binary: int | str = CLASS_MAP[group]
        subject = parts[class_index + 1] if class_index + 1 < len(parts) - 1 else txt_path.parent.name
        batch = parts[class_index - 1] if class_index > 0 else selected_root.name
        group_dir = Path(*parts[: class_index + 1])
    else:
        group = "unknown"
        class_binary = ""
        subject = txt_path.parent.name or "unknown"
        batch = selected_root.name or "batch"
        group_dir = None
    return group, class_binary, subject, batch, group_dir


def _record_for_txt(txt_path: Path, selected_root: Path) -> dict:
    group, class_binary, subject, batch_id, group_dir = _infer_metadata(txt_path, selected_root)
    spectrum_no = _spectrum_number(txt_path.stem)
    try:
        relative_path = txt_path.relative_to(selected_root)
    except ValueError:
        relative_path = Path(txt_path.name)

    tif_candidates: list[Path] = []
    if group_dir is not None:
        tif_candidates = sorted((group_dir.parent / "tif_images" / group).glob(f"subject_{subject}_*.tif"))
    png_candidates = sorted(txt_path.parent.glob("*.png"))
    uid_stem = txt_path.stem
    if group == "unknown":
        uid_stem += "_" + hashlib.sha1(str(txt_path.resolve()).encode("utf-8")).hexdigest()[:8]
    uid_spectrum_no = spectrum_no if group != "unknown" else None
    return {
        "spectrum_uid": _safe_uid(batch_id, group, subject, uid_spectrum_no, uid_stem),
        "batch_id": batch_id,
        "class_original": group,
        "class_binary": class_binary,
        "subject_id": subject,
        "spectrum_no": spectrum_no if spectrum_no is not None else "",
        "file_name": txt_path.name,
        "file_relative_path": str(relative_path),
        "file_absolute_path": str(txt_path.resolve()),
        "selected_folder": str(selected_root),
        "tif_path": str(tif_candidates[0].resolve()) if tif_candidates else "",
        "png_path": str(png_candidates[0].resolve()) if png_candidates else "",
    }


def _sample_one_group(records: list[dict], requested: int, rng: random.Random, strategy: str) -> list[dict]:
    if requested < -1:
        raise ValueError("每组抽样数量必须为-1或非负整数；-1表示该组全部，0表示不抽该组。")
    if requested == 0 or not records:
        return []
    if requested == -1 or requested >= len(records):
        return list(records)
    if strategy == "random":
        return rng.sample(records, requested)
    if strategy != "balanced":
        raise ValueError(f"未知抽样策略：{strategy}")

    subjects: dict[str, list[dict]] = {}
    for record in records:
        subjects.setdefault(str(record.get("subject_id") or "unknown"), []).append(record)
    for queue in subjects.values():
        rng.shuffle(queue)
    subject_order = sorted(subjects, key=_natural_key)
    rng.shuffle(subject_order)
    selected: list[dict] = []
    position = 0
    while len(selected) < requested:
        made_progress = False
        for _ in range(len(subject_order)):
            subject = subject_order[position % len(subject_order)]
            position += 1
            if subjects[subject]:
                selected.append(subjects[subject].pop())
                made_progress = True
                break
        if not made_progress:
            break
    return selected


def sample_records_by_group(
    records: list[dict],
    group0_size: int = 50,
    group1_size: int = 50,
    unknown_size: int = 0,
    seed: int = 20260813,
    strategy: str = "balanced",
) -> list[dict]:
    """按独立配额抽取年轻组(0)、衰老组(1)和未知组光谱。"""
    rng = random.Random(seed)
    pools = {
        0: [record for record in records if record.get("class_binary") == 0],
        1: [record for record in records if record.get("class_binary") == 1],
        "unknown": [record for record in records if record.get("class_binary") not in {0, 1}],
    }
    selected = []
    selected.extend(_sample_one_group(pools[0], group0_size, rng, strategy))
    selected.extend(_sample_one_group(pools[1], group1_size, rng, strategy))
    selected.extend(_sample_one_group(pools["unknown"], unknown_size, rng, strategy))
    if not selected:
        available = {"0(young)": len(pools[0]), "1(aging)": len(pools[1]), "unknown": len(pools["unknown"])}
        raise ValueError(f"抽样结果为空。请检查各组配额；当前可用数量：{available}")

    order_rng = random.Random(seed ^ 0x5A17)
    order_rng.shuffle(selected)
    requested_map = {0: group0_size, 1: group1_size, "unknown": unknown_size}
    for index, record in enumerate(selected, start=1):
        label = record.get("class_binary") if record.get("class_binary") in {0, 1} else "unknown"
        requested = requested_map[label]
        available_count = len(pools[label])
        record["sampling_strategy"] = "all" if requested == -1 or requested >= available_count else strategy
        record["sampling_seed"] = seed
        record["sample_order"] = index
    return selected


def sample_records(records: list[dict], sample_size: int = 0, seed: int = 20260813, strategy: str = "balanced") -> list[dict]:
    """Return a reproducible full/random/balanced spectrum sample.

    Balanced sampling alternates class groups and rotates through subjects
    within each class. This prevents the 15 aging subjects from dominating the
    five young subjects merely because more aging spectra are available.
    """
    if sample_size < 0:
        raise ValueError("抽样数量不能为负数；0表示使用全部光谱。")
    if sample_size == 0 or sample_size >= len(records):
        selected = list(records)
    else:
        rng = random.Random(seed)
        if strategy == "random":
            selected = rng.sample(records, sample_size)
        elif strategy == "balanced":
            pools: dict[str, dict[str, list[dict]]] = {}
            for record in records:
                group = str(record.get("class_original") or "unknown")
                subject = str(record.get("subject_id") or "unknown")
                pools.setdefault(group, {}).setdefault(subject, []).append(record)
            for subjects in pools.values():
                for queue in subjects.values():
                    rng.shuffle(queue)

            groups = sorted(pools, key=_natural_key)
            rng.shuffle(groups)
            subject_orders = {group: sorted(pools[group], key=_natural_key) for group in groups}
            for order in subject_orders.values():
                rng.shuffle(order)
            subject_positions = {group: 0 for group in groups}
            selected = []
            while len(selected) < sample_size:
                made_progress = False
                for group in groups:
                    subjects = subject_orders[group]
                    for _ in range(len(subjects)):
                        position = subject_positions[group] % len(subjects)
                        subject_positions[group] += 1
                        subject = subjects[position]
                        if pools[group][subject]:
                            selected.append(pools[group][subject].pop())
                            made_progress = True
                            break
                    if len(selected) >= sample_size:
                        break
                if not made_progress:
                    break
        else:
            raise ValueError(f"未知抽样策略：{strategy}")

    order_rng = random.Random(seed ^ 0x5A17)
    order_rng.shuffle(selected)
    for index, record in enumerate(selected, start=1):
        record["sampling_strategy"] = "all" if sample_size == 0 or sample_size >= len(records) else strategy
        record["sampling_seed"] = seed
        record["sample_order"] = index
    return selected


def scan_spectra(
    data_root: str | Path,
    blind_order: bool = True,
    sample_size: int = 0,
    seed: int = 20260813,
    strategy: str = "balanced",
    group0_size: int | None = None,
    group1_size: int | None = None,
    unknown_size: int = 0,
) -> list[dict]:
    """递归扫描任意所选文件夹下的TXT光谱并按设置抽样。"""
    root = resolve_input_folder(data_root)
    txt_files = sorted((path for path in root.rglob("*.txt") if path.is_file()), key=lambda path: _natural_key(str(path)))
    records = [_record_for_txt(path, root) for path in txt_files]
    if not records:
        raise ValueError(f"所选文件夹及其子文件夹中没有找到TXT：{root}")
    if group0_size is not None or group1_size is not None:
        return sample_records_by_group(
            records,
            group0_size=50 if group0_size is None else group0_size,
            group1_size=50 if group1_size is None else group1_size,
            unknown_size=unknown_size,
            seed=seed,
            strategy=strategy,
        )
    if blind_order or sample_size:
        return sample_records(records, sample_size=sample_size, seed=seed, strategy=strategy)
    for index, record in enumerate(records, start=1):
        record.update({"sampling_strategy": "all", "sampling_seed": seed, "sample_order": index})
    return records


def read_spectrum(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        values = np.loadtxt(path, dtype=float)
    except Exception as exc:
        raise ValueError(f"Cannot read numeric spectrum: {path}\n{exc}") from exc

    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Spectrum must contain exactly two numeric columns: {path}")
    return pd.DataFrame(values, columns=["Raman_Shift", "Raman_Intensity"])


def _longest_equal_run(values: np.ndarray) -> int:
    if len(values) == 0:
        return 0
    longest = current = 1
    for index in range(1, len(values)):
        if values[index] == values[index - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return int(longest)


def smoothed_intensity(intensity: Iterable[float]) -> np.ndarray:
    y = np.asarray(list(intensity), dtype=float)
    if len(y) < 7:
        return y.copy()
    window = min(21, len(y) if len(y) % 2 else len(y) - 1)
    window = max(window, 5)
    return savgol_filter(y, window_length=window, polyorder=min(3, window - 1))


def calculate_qc_metrics(spectrum: pd.DataFrame) -> dict:

    x = spectrum["Raman_Shift"].to_numpy(dtype=float)
    y = spectrum["Raman_Intensity"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    missing_count = int((~finite).sum())
    xf, yf = x[finite], y[finite]
    if len(xf) == 0:
        raise ValueError("Spectrum has no finite numeric points")

    smooth = smoothed_intensity(yf)
    residual = yf - smooth
    residual_median = np.median(residual)
    residual_mad = np.median(np.abs(residual - residual_median))
    robust_noise = 1.4826 * residual_mad
    signal_rms = float(np.sqrt(np.mean((smooth - np.median(smooth)) ** 2)))
    robust_snr_db = 20 * np.log10(signal_rms / robust_noise) if robust_noise > 0 and signal_rms > 0 else np.inf
    spike_threshold = 8 * robust_noise
    spike_count = int(np.sum(np.abs(residual - residual_median) > spike_threshold)) if spike_threshold > 0 else 0

    p05, intensity_median, p95 = np.percentile(yf, [5, 50, 95])

    def region_values(x_min: float, x_max: float) -> tuple[np.ndarray, np.ndarray]:
        mask = (xf >= x_min) & (xf <= x_max)
        return smooth[mask], residual[mask]

    fingerprint_smooth, _ = region_values(500, 1800)
    _, silent_residual = region_values(1800, 2700)
    ch_smooth, _ = region_values(2700, 3200)

    def robust_contrast(values: np.ndarray) -> float:
        if len(values) == 0:
            return np.nan
        low, high = np.percentile(values, [20, 95])
        return float(high - low)

    if len(silent_residual):
        silent_center = np.median(silent_residual)
        silent_noise_mad = float(1.4826 * np.median(np.abs(silent_residual - silent_center)))
    else:
        silent_noise_mad = np.nan

    return {
        "n_points": int(len(spectrum)),
        "x_min": float(np.min(xf)),
        "x_max": float(np.max(xf)),
        "x_step_median": float(np.median(np.diff(xf))) if len(xf) > 1 else np.nan,
        "intensity_min": float(np.min(yf)),
        "intensity_max": float(np.max(yf)),
        "intensity_mean": float(np.mean(yf)),
        "intensity_std": float(np.std(yf)),
        "intensity_p05": float(p05),
        "intensity_median": float(intensity_median),
        "intensity_p95": float(p95),
        "robust_intensity_range": float(p95 - p05),
        "robust_snr_db": float(robust_snr_db),
        "silent_noise_mad": silent_noise_mad,
        "fingerprint_contrast": robust_contrast(fingerprint_smooth),
        "ch_contrast": robust_contrast(ch_smooth),
        "spike_count_auto": spike_count,
        "saturation_run_max": _longest_equal_run(yf),
        "axis_monotonic": int(bool(len(xf) < 2 or np.all(np.diff(xf) > 0))),
        "missing_count": missing_count,
        "finite_fraction": float(finite.mean()) if len(finite) else 0.0,
    }


def empty_annotations() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def sample_manifest_path(annotation_csv: str | Path) -> Path:
    path = Path(annotation_csv).expanduser().resolve()
    return path.with_name(path.stem + "_sample_manifest.csv")


def save_sample_manifest(records: list[dict], annotation_csv: str | Path) -> Path:
   
    path = sample_manifest_path(annotation_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(records, columns=SAMPLE_MANIFEST_COLUMNS).to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)
    return path


def load_sample_manifest(annotation_csv: str | Path) -> list[dict]:
    path = sample_manifest_path(annotation_csv)
    if not path.exists():
        return []
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"subject_id": str})
    missing = [column for column in SAMPLE_MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"抽样清单缺少字段：{', '.join(missing)}")
    aging_wrong = (frame["class_original"].astype(str).str.lower() == "aging") & (pd.to_numeric(frame["class_binary"], errors="coerce") != 1)
    young_wrong = (frame["class_original"].astype(str).str.lower() == "young") & (pd.to_numeric(frame["class_binary"], errors="coerce") != 0)
    if aging_wrong.any() or young_wrong.any():
        raise ValueError(
            "该抽样清单使用旧的相反编码。当前统一规定为0=young（年轻）、1=aging（衰老）。"
            "请新建一个标注CSV文件名重新抽样，避免混用标签。"
        )
    frame = frame.sort_values("sample_order", kind="stable")
    records = frame[SAMPLE_MANIFEST_COLUMNS].fillna("").to_dict("records")
    absent = [row["file_absolute_path"] for row in records if not Path(str(row["file_absolute_path"])).is_file()]
    if absent:
        raise ValueError(f"抽样清单中的原始TXT已不存在，例如：{absent[0]}")
    return records


def load_annotations(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        return empty_annotations()
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"subject_id": str})
    for column in OUTPUT_COLUMNS:
        if column not in frame.columns:
            if column in METRIC_FIELDS:
               
                frame[column] = np.nan
            else:
                frame[column] = "" if column in {"other_mark", "qc_status"} else 0
    aging_wrong = (frame["class_original"].astype(str).str.lower() == "aging") & (pd.to_numeric(frame["class_binary"], errors="coerce") != 1)
    young_wrong = (frame["class_original"].astype(str).str.lower() == "young") & (pd.to_numeric(frame["class_binary"], errors="coerce") != 0)
    if aging_wrong.any() or young_wrong.any():
        raise ValueError(
            "该标注CSV使用旧的相反编码。当前统一规定为0=young（年轻）、1=aging（衰老）。"
            "请改用新的标注CSV文件名。"
        )
    return frame[OUTPUT_COLUMNS].copy()


def annotation_for_path(annotations: pd.DataFrame, file_path: str) -> dict | None:
    if annotations is None or annotations.empty:
        return None
    matched = annotations[annotations["file_absolute_path"].astype(str) == str(file_path)]
    return None if matched.empty else matched.iloc[-1].to_dict()


def build_annotation(
    record: dict,
    metrics: dict,
    qc_status: str,
    issues: dict[str, bool],
    notes: str = "",
) -> dict:

    qc_status = qc_status.strip().lower()
    notes = notes.strip()
    if qc_status not in QC_STATUSES:
        raise ValueError("Choose one QC status: pass, review, or fail")
    selected = [name for name in ISSUE_FIELDS if bool(issues.get(name, False))]
    if qc_status == "pass" and selected:
        raise ValueError("A pass spectrum cannot also have an issue selected")
    if qc_status in {"review", "fail"} and not selected:
        raise ValueError("REVIEW/FAIL至少需要勾选一个异常原因")

    row = {column: "" for column in OUTPUT_COLUMNS}
    row.update(record)
    row.update(metrics)
    row["qc_status"] = qc_status
    for field in ISSUE_FIELDS:
        row[field] = int(bool(issues.get(field, False)))
    row["other_mark"] = notes
    row["annotated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    row["tool_version"] = TOOL_VERSION
    return row


def upsert_annotation(annotations: pd.DataFrame, row: dict) -> pd.DataFrame:
    frame = annotations.copy() if annotations is not None else empty_annotations()
    if not frame.empty:
        frame = frame[frame["file_absolute_path"].astype(str) != str(row["file_absolute_path"])].copy()
    frame = pd.concat([frame, pd.DataFrame([row], columns=OUTPUT_COLUMNS)], ignore_index=True)
    return frame[OUTPUT_COLUMNS]


def save_annotations(annotations: pd.DataFrame, csv_path: str | Path) -> None:
    path = Path(csv_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    annotations.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def summarize_records(records: list[dict]) -> pd.DataFrame:
    return (
        pd.DataFrame(records)
        .groupby(["class_original", "class_binary", "subject_id"], dropna=False)
        .size()
        .rename("spectrum_count")
        .reset_index()
    )
