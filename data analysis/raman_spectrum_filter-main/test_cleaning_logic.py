import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core import Back_Filter_Excute, _passes_maximum, _passes_minimum
from parameter_finding import Filter_Data_Collect, Index_Comparison
from state import Parameter_State


class CleaningLogicTests(unittest.TestCase):
    def test_zero_limit_disables_missing_peak_condition(self):
        self.assertTrue(_passes_minimum(np.nan, 0))
        self.assertTrue(_passes_maximum(np.nan, 0))
        self.assertTrue(_passes_maximum(np.nan, 10000))
        self.assertFalse(_passes_minimum(np.nan, 1))
        self.assertFalse(_passes_maximum(np.nan, 1))

    def test_spectrum_without_local_peaks_still_returns_technical_metrics(self):
        x = np.arange(500.0, 3601.0, 3.0)
        y = 1000.0 + 0.02 * x
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "monotonic.txt"
            np.savetxt(path, np.column_stack([x, y]), delimiter="\t")
            frame = pd.DataFrame({"Raman_Shift": x, "Raman_Intensity": y})
            params = Parameter_State()
            finger = Index_Comparison(frame, params.Finger_Start, params.Finger_End)
            silence = Index_Comparison(frame, params.Silence_Start, params.Silence_End)
            ch = Index_Comparison(frame, params.CH_Start, params.CH_End)
            result = Filter_Data_Collect(params, path, *finger, *silence, *ch)

        self.assertTrue(np.isfinite(result["Finger_Peak_STD"]))
        self.assertTrue(np.isfinite(result["CH_Peak_STD"]))
        self.assertTrue(np.isnan(result["Peak_Height_Ratio_3"]))

    def test_cleaning_creates_independent_run_and_manifest(self):
        x = np.arange(500.0, 3601.0, 3.0)
        y = 1000.0 + 0.02 * x
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            input_root = root / "input" / "young" / "01"
            output_root = root / "output"
            input_root.mkdir(parents=True)
            output_root.mkdir()
            source = input_root / "spectrum.txt"
            np.savetxt(source, np.column_stack([x, y]), delimiter="\t")
            old_result = output_root / "old_result.txt"
            old_result.write_text("must remain", encoding="utf-8")

            params = Parameter_State(
                Input_Folder_Path=str(root / "input"),
                Output_Folder_Path=str(output_root),
            )
            first = Back_Filter_Excute(params, lambda _: None, lambda _: None)
            second = Back_Filter_Excute(params, lambda _: None, lambda _: None)

            self.assertNotEqual(first, second)
            self.assertTrue(old_result.exists())
            self.assertTrue((first / "passed" / "young" / "01" / "spectrum.txt").is_file())
            manifest = pd.read_csv(first / "run_manifest.csv", encoding="utf-8-sig")
            self.assertEqual(manifest.loc[0, "status"], "pass")
            self.assertEqual(manifest.loc[0, "stage"], "cleaning")


if __name__ == "__main__":
    unittest.main()
