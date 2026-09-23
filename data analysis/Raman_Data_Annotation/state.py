
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class ProcessState:
    data_root: Path = BASE_DIR / "raw_data"
    save_csv_path: Path = BASE_DIR / "result" / "qc_annotations.csv"
    group0_size: int = 50
    group1_size: int = 50
    unknown_size: int = 0
    sample_seed: int = 20260813
    sampling_strategy: str = "balanced"
    records: list[dict] = field(default_factory=list)
    current_index: int = 0
    annotations: Optional[pd.DataFrame] = None
    current_spectrum: Optional[pd.DataFrame] = None
    current_metrics: dict = field(default_factory=dict)
