import unittest

import pandas as pd

from qc_assistant import predict_qc, train_qc_tree


def feature_row(status: str, good: bool, index: int) -> dict:
    return {
        "qc_status": status,
        "n_points": 1340,
        "x_min": 273.1,
        "x_max": 3746.9,
        "x_step_median": 2.59,
        "intensity_std": 220 if good else 40,
        "intensity_p05": 1200,
        "intensity_median": 1500,
        "intensity_p95": 2200 if good else 1550,
        "robust_intensity_range": 1000 if good else 350,
        "robust_snr_db": 28 + index * 0.01 if good else 5 + index * 0.01,
        "silent_noise_mad": 12 if good else 90,
        "fingerprint_contrast": 300 if good else 30,
        "ch_contrast": 900 if good else 80,
        "spike_count_auto": 0 if good else 12,
        "saturation_run_max": 1,
        "axis_monotonic": 1,
        "missing_count": 0,
        "finite_fraction": 1.0,
    }


class QCAssistantTests(unittest.TestCase):
    def test_does_not_train_with_too_few_labels(self):
        frame = pd.DataFrame([feature_row("pass", True, index) for index in range(10)])
        model, message = train_qc_tree(frame)
        self.assertIsNone(model)
        self.assertIn("待训练", message)

    def test_tree_suggests_pass_and_attention(self):
        rows = [feature_row("pass", True, index) for index in range(20)]
        rows += [feature_row("fail", False, index) for index in range(20)]
        model, message = train_qc_tree(pd.DataFrame(rows))
        self.assertIsNotNone(model)
        self.assertIn("已训练", message)

        good = predict_qc(model, feature_row("", True, 99))
        bad = predict_qc(model, feature_row("", False, 99))
        self.assertEqual(good["suggestion"], "pass")
        self.assertEqual(bad["suggestion"], "attention")


if __name__ == "__main__":
    unittest.main()
