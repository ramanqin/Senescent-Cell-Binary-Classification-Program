import tempfile
import threading
import unittest
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np

from app import SpectrumPreprocessorApp
from spectral_preprocessor import PreprocessConfig, find_spectra


class FileDiscoveryTests(unittest.TestCase):
    def test_reports_failed_and_result_directories_are_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            included = root / "young" / "01" / "spectrum.txt"
            included.parent.mkdir(parents=True)
            included.write_text("600\t1\n603\t2\n", encoding="utf-8")
            for path in (
                root / "failed" / "young" / "bad.txt",
                root / "result_plot" / "predictions.csv",
                root / "run_manifest.csv",
                root / "处理报告_20260101.csv",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("600,1\n603,2\n", encoding="utf-8")

            found = find_spectra(root, ["txt", "csv"], recursive=True)
            self.assertEqual(found, [included])

    def test_batch_processing_creates_versioned_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            input_root = root / "cleaning_run_test" / "passed"
            source = input_root / "young" / "01" / "spectrum.txt"
            source.parent.mkdir(parents=True)
            x = np.arange(600.0, 1801.0, 3.0)
            y = 1000.0 + np.sin(x / 80.0)
            np.savetxt(source, np.column_stack([x, y]), delimiter="\t")
            output_root = root / "preprocessed"
            dummy = SimpleNamespace(stop_event=threading.Event(), events=Queue())
            options = {
                "format": "txt",
                "suffix": "_processed",
                "precision": 8,
                "preserve": True,
                "overwrite": False,
            }

            SpectrumPreprocessorApp._batch_worker(
                dummy, input_root, output_root, [source], PreprocessConfig(), options
            )

            runs = list(output_root.glob("preprocessing_run_*"))
            self.assertEqual(len(runs), 1)
            manifest = (runs[0] / "run_manifest.csv").read_text(encoding="utf-8-sig")
            self.assertIn("upstream_run_id", manifest)
            self.assertIn("success", manifest)
            self.assertTrue((runs[0] / "young" / "01" / "spectrum_processed.txt").is_file())


if __name__ == "__main__":
    unittest.main()
