"""Generate tiny synthetic Raman spectra for checking the project's file flow.

These are deliberately constructed signals, not experimental observations.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "data analysis" / "Spectral Batch Processing"))
from spectral_preprocessor import PreprocessConfig, preprocess, save_spectrum  # noqa: E402


def gaussian(x: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - center) / width) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成仅用于流程验收的合成拉曼光谱")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="空目录；将在其中建立原始光谱和已预处理光谱")
    args = parser.parse_args()
    root = args.output_root.resolve()
    raw_root = root / "原始光谱"
    processed_root = root / "已预处理光谱"
    if raw_root.exists() or processed_root.exists():
        parser.error(f"演示光谱目录已存在，拒绝覆盖：{root}")
    root.mkdir(parents=True, exist_ok=True)
    x = np.arange(500.0, 3301.0, 2.0)
    cfg = PreprocessConfig()
    cfg.validate()
    rng = np.random.default_rng(20260923)

    for label in ("年轻", "衰老"):
        for number in range(1, 6):
            for replicate in range(1, 3):
                individual = 1.0 + 0.015 * (number - 3)
                broad = 82 + 0.012 * (x - 500) + 5 * np.sin(x / 340)
                fingerprint = (
                    (110 if label == "年轻" else 78) * gaussian(x, 730, 14)
                    + 95 * gaussian(x, 1000, 22)
                    + (75 if label == "年轻" else 112) * gaussian(x, 1450, 26)
                    + 60 * gaussian(x, 1650, 20)
                )
                silent = 90 * gaussian(x, 2250, 40)
                ch = 120 * gaussian(x, 2930, 55)
                y = broad + individual * (fingerprint + silent + ch)
                y += rng.normal(0, 0.12, size=x.size)
                raw = raw_root / label / f"编号{number:02d}" / f"重复{replicate}.txt"
                raw.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(raw, np.column_stack([x, y]), fmt="%.8f", delimiter="\t")

                result = preprocess(x, y, cfg)
                processed = processed_root / label / f"编号{number:02d}" / f"重复{replicate}_预处理.txt"
                save_spectrum(processed, result.x, result.y, fmt="txt", precision=8)
    print(f"已生成：{raw_root} 与 {processed_root}（均为合成演示数据）")


if __name__ == "__main__":
    main()
