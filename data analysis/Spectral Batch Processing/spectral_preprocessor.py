from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.integrate import trapezoid
from scipy.signal import medfilt, savgol_filter
from scipy.sparse.linalg import spsolve


@dataclass
class PreprocessConfig:
    # 波数范围，可写成“600-1800,2700-3600”
    crop_enabled: bool = True
    ranges: str = "600-1800"

    # 宇宙射线：局部中位数残差/MAD
    cosmic_enabled: bool = True
    cosmic_window: int = 5
    cosmic_threshold: float = 12.0
    cosmic_max_width: int = 3
    cosmic_passes: int = 2

    # 重采样，每个保留波段分别插值，避免跨越静默区
    resample_enabled: bool = True
    resample_step: float = 3.0

    # 基线
    baseline_method: str = "ALS"  # 无、ALS、airPLS、多项式
    baseline_lambda: float = 100000.0
    baseline_p: float = 0.01
    baseline_iterations: int = 15
    baseline_poly_order: int = 5

    # SG平滑/导数
    sg_enabled: bool = True
    sg_window: int = 7
    sg_polyorder: int = 2
    sg_derivative: int = 0

    # 归一化：无、向量归一化、面积归一化、SNV、Min-Max
    normalization: str = "向量归一化"

    # 数据质量
    min_points: int = 20

    def validate(self) -> None:
        if self.crop_enabled:
            parse_ranges(self.ranges)
        if self.cosmic_window < 3 or self.cosmic_window % 2 == 0:
            raise ValueError("宇宙射线窗口必须是≥3的奇数")
        if self.cosmic_threshold <= 0:
            raise ValueError("宇宙射线阈值必须大于0")
        if self.cosmic_max_width < 1 or self.cosmic_passes < 1:
            raise ValueError("宇宙射线最大宽度和迭代次数必须≥1")
        if self.resample_enabled and self.resample_step <= 0:
            raise ValueError("重采样步长必须大于0")
        if self.baseline_lambda <= 0:
            raise ValueError("基线λ必须大于0")
        if not 0 < self.baseline_p < 1:
            raise ValueError("ALS不对称参数p必须在0与1之间")
        if self.baseline_iterations < 1:
            raise ValueError("基线迭代次数必须≥1")
        if self.baseline_poly_order < 1:
            raise ValueError("多项式阶数必须≥1")
        if self.sg_enabled:
            if self.sg_window < 3 or self.sg_window % 2 == 0:
                raise ValueError("SG窗口必须是≥3的奇数")
            if self.sg_polyorder >= self.sg_window:
                raise ValueError("SG多项式阶数必须小于窗口")
            if self.sg_derivative < 0 or self.sg_derivative > self.sg_polyorder:
                raise ValueError("SG导数阶数必须在0与多项式阶数之间")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PreprocessConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        values = payload.get("preprocess", payload)
        return cls(**values)


@dataclass
class ProcessResult:
    x: np.ndarray
    y: np.ndarray
    baseline: np.ndarray | None
    cosmic_points: int
    input_points: int
    output_points: int


def parse_ranges(text: str) -> list[tuple[float, float]]:
    parts = [p.strip() for p in re.split(r"[,，;；]+", text) if p.strip()]
    result: list[tuple[float, float]] = []
    for part in parts:
        m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*[-~—–至]\s*(-?\d+(?:\.\d+)?)\s*", part)
        if not m:
            raise ValueError(f"无法识别波数范围：{part}；示例：600-1800,2700-3600")
        start, end = map(float, m.groups())
        if start >= end:
            raise ValueError(f"波数范围起点必须小于终点：{part}")
        result.append((start, end))
    if not result:
        raise ValueError("至少需要一个波数范围")
    result.sort()
    for (_, prev_end), (start, _) in zip(result, result[1:]):
        if start <= prev_end:
            raise ValueError("波数范围不能重叠")
    return result


def read_spectrum(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读取TXT/CSV/DAT；自动跳过表头和非数值行，使用前两列数值。"""
    xs: list[float] = []
    ys: list[float] = []
    with Path(path).open("r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            fields = [v for v in re.split(r"[\t,;\s]+", line.strip()) if v]
            if len(fields) < 2:
                continue
            try:
                xs.append(float(fields[0]))
                ys.append(float(fields[1]))
            except ValueError:
                continue
    if len(xs) < 2:
        raise ValueError("未读取到至少两行双列数值数据")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    order = np.argsort(x)
    x, y = x[order], y[order]
    # 合并重复波数点
    unique_x, inverse = np.unique(x, return_inverse=True)
    if len(unique_x) != len(x):
        sums = np.bincount(inverse, weights=y)
        counts = np.bincount(inverse)
        x, y = unique_x, sums / counts
    return x, y


def _segments(x: np.ndarray) -> list[slice]:
    if len(x) < 3:
        return [slice(0, len(x))]
    dx = np.diff(x)
    typical = np.median(dx[dx > 0])
    cuts = np.where(dx > max(typical * 3.0, typical + 1e-9))[0] + 1
    bounds = np.r_[0, cuts, len(x)]
    return [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


def remove_cosmic_rays(
    y: np.ndarray, window: int, threshold: float, max_width: int, passes: int
) -> tuple[np.ndarray, int]:
    result = np.asarray(y, dtype=float).copy()
    corrected: set[int] = set()
    for _ in range(passes):
        local = medfilt(result, kernel_size=window)
        residual = result - local
        center = np.median(residual)
        sigma = 1.4826 * np.median(np.abs(residual - center))
        if not np.isfinite(sigma) or sigma <= np.finfo(float).eps:
            break
        candidates = np.flatnonzero(np.abs(residual - center) > threshold * sigma)
        if not len(candidates):
            break
        groups = np.split(candidates, np.where(np.diff(candidates) > 1)[0] + 1)
        changed = False
        for group in groups:
            if not 1 <= len(group) <= max_width:
                continue
            left, right = int(group[0] - 1), int(group[-1] + 1)
            if left < 0 or right >= len(result):
                continue
            result[group] = np.interp(group, [left, right], [result[left], result[right]])
            corrected.update(map(int, group))
            changed = True
        if not changed:
            break
    return result, len(corrected)


def baseline_als(y: np.ndarray, lam: float, p: float, iterations: int) -> np.ndarray:
    n = len(y)
    d = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    penalty = lam * (d.T @ d)
    w = np.ones(n)
    for _ in range(iterations):
        z = spsolve((sparse.diags(w, 0) + penalty).tocsc(), w * y)
        w = p * (y > z) + (1.0 - p) * (y <= z)
    return np.asarray(z)


def baseline_airpls(y: np.ndarray, lam: float, iterations: int) -> np.ndarray:
    n = len(y)
    d = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    penalty = lam * (d.T @ d)
    w = np.ones(n)
    z = np.zeros(n)
    for i in range(1, iterations + 1):
        z = spsolve((sparse.diags(w, 0) + penalty).tocsc(), w * y)
        residual = y - z
        negative = residual[residual < 0]
        total = abs(negative.sum())
        if total < 1e-3 * np.sum(np.abs(y)) or not len(negative):
            break
        w[residual >= 0] = 0
        w[residual < 0] = np.exp(np.clip(i * np.abs(residual[residual < 0]) / total, 0, 50))
        edge_weight = float(np.exp(np.clip(i * np.max(np.abs(negative)) / total, 0, 50)))
        w[0] = w[-1] = edge_weight
    return np.asarray(z)


def polynomial_baseline(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    scaled = 2.0 * (x - x.min()) / max(x.max() - x.min(), np.finfo(float).eps) - 1.0
    coeff = np.polyfit(scaled, y, order)
    return np.polyval(coeff, scaled)


def normalize(y: np.ndarray, method: str, x: np.ndarray) -> np.ndarray:
    eps = np.finfo(float).eps
    if method == "无":
        return y
    if method == "向量归一化":
        scale = np.linalg.norm(y)
        return y / scale if scale > eps else y
    if method == "面积归一化":
        scale = abs(trapezoid(np.abs(y), x))
        return y / scale if scale > eps else y
    if method == "SNV":
        std = np.std(y, ddof=1)
        return (y - np.mean(y)) / std if std > eps else y - np.mean(y)
    if method == "Min-Max":
        span = np.max(y) - np.min(y)
        return (y - np.min(y)) / span if span > eps else y - np.min(y)
    raise ValueError(f"未知归一化方法：{method}")


def _apply_by_segment(x: np.ndarray, y: np.ndarray, func) -> np.ndarray:
    out = np.empty_like(y, dtype=float)
    for segment in _segments(x):
        out[segment] = func(x[segment], y[segment])
    return out


def preprocess(x: np.ndarray, y: np.ndarray, cfg: PreprocessConfig) -> ProcessResult:
    cfg.validate()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    input_points = len(x)
    if len(x) < cfg.min_points:
        raise ValueError(f"有效点数{len(x)}少于最低要求{cfg.min_points}")

    cosmic_points = 0
    if cfg.cosmic_enabled:
        y, cosmic_points = remove_cosmic_rays(
            y, cfg.cosmic_window, cfg.cosmic_threshold, cfg.cosmic_max_width, cfg.cosmic_passes
        )

    ranges = parse_ranges(cfg.ranges) if cfg.crop_enabled else [(float(x.min()), float(x.max()))]
    pieces_x: list[np.ndarray] = []
    pieces_y: list[np.ndarray] = []
    for start, end in ranges:
        if cfg.resample_enabled:
            grid = np.arange(start, end + cfg.resample_step * 0.25, cfg.resample_step)
            grid = grid[(grid >= x.min()) & (grid <= x.max())]
            py = np.interp(grid, x, y)
            px = grid
        else:
            mask = (x >= start) & (x <= end)
            px, py = x[mask], y[mask]
        if len(px) < cfg.min_points:
            raise ValueError(f"波段{start:g}-{end:g} cm⁻¹只有{len(px)}个点")
        pieces_x.append(px)
        pieces_y.append(py)
    x = np.concatenate(pieces_x)
    y = np.concatenate(pieces_y)

    baseline_total = np.zeros_like(y)
    method = cfg.baseline_method
    if method != "无":
        for segment in _segments(x):
            sx, sy = x[segment], y[segment]
            if method == "ALS":
                base = baseline_als(sy, cfg.baseline_lambda, cfg.baseline_p, cfg.baseline_iterations)
            elif method == "airPLS":
                base = baseline_airpls(sy, cfg.baseline_lambda, cfg.baseline_iterations)
            elif method == "多项式":
                base = polynomial_baseline(sx, sy, cfg.baseline_poly_order)
            else:
                raise ValueError(f"未知基线方法：{method}")
            baseline_total[segment] = base
        y = y - baseline_total
    else:
        baseline_total = None

    if cfg.sg_enabled:
        def smooth(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
            if len(sy) < cfg.sg_window:
                raise ValueError(f"光谱点数小于SG窗口{cfg.sg_window}")
            delta = float(np.median(np.diff(sx))) if len(sx) > 1 else 1.0
            return savgol_filter(
                sy, cfg.sg_window, cfg.sg_polyorder, deriv=cfg.sg_derivative,
                delta=delta, mode="interp"
            )
        y = _apply_by_segment(x, y, smooth)

    y = normalize(y, cfg.normalization, x)
    return ProcessResult(x=x, y=y, baseline=baseline_total, cosmic_points=cosmic_points,
                         input_points=input_points, output_points=len(x))


def save_spectrum(path: str | Path, x: np.ndarray, y: np.ndarray, fmt: str = "txt", precision: int = 8) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    delimiter = "," if fmt.lower() == "csv" else "\t"
    header = "Raman_shift_cm-1,Intensity" if fmt.lower() == "csv" else "Raman_shift_cm-1\tIntensity"
    np.savetxt(path, np.column_stack([x, y]), delimiter=delimiter,
               fmt=f"%.{precision}g", header=header, comments="")


def find_spectra(root: str | Path, extensions: Iterable[str], recursive: bool = True) -> list[Path]:
    root = Path(root)
    wanted = {e.lower().lstrip(".") for e in extensions}
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower().lstrip(".") in wanted)
