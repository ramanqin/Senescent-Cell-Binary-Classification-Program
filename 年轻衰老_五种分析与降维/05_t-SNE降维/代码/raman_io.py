from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SpectralDataset:
    wavenumbers: np.ndarray
    matrix: np.ndarray
    labels: np.ndarray
    sample_ids: list[str]
    spectra_counts: list[int]
    class_names: tuple[str, str]


def read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the first two numeric columns of a text spectrum."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "gbk", "gb2312"):
        try:
            frame = pd.read_csv(
                path,
                sep=None,
                engine="python",
                header=None,
                encoding=encoding,
                comment="#",
            )
            if frame.shape[1] < 2:
                raise ValueError("fewer than two columns")
            x = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
            keep = np.isfinite(x) & np.isfinite(y)
            x, y = x[keep], y[keep]
            order = np.argsort(x)
            x, y = x[order], y[order]
            unique_x, inverse = np.unique(x, return_inverse=True)
            if len(unique_x) != len(x):
                y = np.bincount(inverse, weights=y) / np.bincount(inverse)
                x = unique_x
            if len(x) < 20 or np.any(np.diff(x) <= 0):
                raise ValueError("invalid Raman-shift grid")
            return x, y
        except Exception as error:  # try the next common encoding
            last_error = error
    raise ValueError(f"Could not read spectrum {path}: {last_error}")


def load_subject_dataset(
    input_dir: str | Path,
    negative_class: str = "年轻",
    positive_class: str = "衰老",
    grid_tolerance: float = 1e-6,
) -> SpectralDataset:
    """Load class/subject/*.txt spectra and average replicates within subjects."""
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    reference_x: np.ndarray | None = None
    subject_spectra: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    spectra_counts: list[int] = []

    for class_code, class_name in enumerate((negative_class, positive_class)):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class directory does not exist: {class_dir}")
        # Stable lexicographic order matches the original analysis manifests and
        # matters for exact reproduction of stochastic t-SNE layouts.
        subject_dirs = sorted(
            (path for path in class_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name.casefold(),
        )
        if not subject_dirs:
            raise ValueError(f"No subject directories found under {class_dir}")

        for subject_dir in subject_dirs:
            files = sorted(subject_dir.glob("*.txt"))
            if not files:
                raise ValueError(f"No TXT spectra found under {subject_dir}")
            spectra = []
            for path in files:
                x, y = read_spectrum(path)
                if reference_x is None:
                    reference_x = x
                if len(x) != len(reference_x) or not np.allclose(
                    x, reference_x, rtol=0, atol=grid_tolerance
                ):
                    raise ValueError(
                        f"Raman-shift grid mismatch: {path}. Resample all spectra first."
                    )
                spectra.append(y)
            subject_spectra.append(np.mean(np.vstack(spectra), axis=0))
            labels.append(class_code)
            sample_ids.append(f"{class_name}/{subject_dir.name}")
            spectra_counts.append(len(spectra))

    assert reference_x is not None
    matrix = np.vstack(subject_spectra)
    label_array = np.asarray(labels, dtype=int)
    class_counts = np.bincount(label_array, minlength=2)
    if class_counts.min() < 5:
        raise ValueError("Each class must contain at least five independent subjects.")
    return SpectralDataset(
        wavenumbers=reference_x,
        matrix=matrix,
        labels=label_array,
        sample_ids=sample_ids,
        spectra_counts=spectra_counts,
        class_names=(negative_class, positive_class),
    )
