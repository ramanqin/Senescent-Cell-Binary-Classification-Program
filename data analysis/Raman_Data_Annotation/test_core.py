import tempfile
import unittest
from pathlib import Path

import numpy as np

from core import (
    ISSUE_FIELDS,
    build_annotation,
    calculate_qc_metrics,
    read_spectrum,
    resolve_input_folder,
    sample_records,
    sample_records_by_group,
    load_sample_manifest,
    save_sample_manifest,
    scan_spectra,
)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "raw_data"
        for group, subject in [("young", "11"), ("aging", "2")]:
            folder = self.root / group / subject
            folder.mkdir(parents=True)
            x = np.linspace(273.14, 3746.91, 1340)
            y = 1000 + 100 * np.sin(x / 150)
            np.savetxt(folder / "Auto-1_spec.txt", np.column_stack([x, y]), delimiter="\t")

    def tearDown(self):
        self.temp.cleanup()

    def test_scan_and_labels(self):
        records = scan_spectra(self.root, blind_order=False)
        self.assertEqual(len(records), 2)
        labels = {r["class_original"]: r["class_binary"] for r in records}
        self.assertEqual(labels, {"aging": 1, "young": 0})

    def test_any_folder_is_accepted(self):
        self.assertEqual(resolve_input_folder(self.root / "young"), (self.root / "young").resolve())
        self.assertEqual(resolve_input_folder(self.root / "young" / "11"), (self.root / "young" / "11").resolve())

    def test_group_and_subject_folder_scans(self):
        self.assertEqual(len(scan_spectra(self.root / "young", blind_order=False)), 1)
        subject_records = scan_spectra(self.root / "aging" / "2", blind_order=False)
        self.assertEqual(len(subject_records), 1)
        self.assertEqual(subject_records[0]["class_original"], "aging")

    def test_reproducible_balanced_sampling(self):
        records = []
        for group in ("young", "aging"):
            for subject in ("1", "2"):
                for number in range(10):
                    records.append({"class_original": group, "subject_id": subject, "spectrum_uid": f"{group}-{subject}-{number}"})
        first = sample_records([dict(row) for row in records], 12, seed=42, strategy="balanced")
        second = sample_records([dict(row) for row in records], 12, seed=42, strategy="balanced")
        self.assertEqual([row["spectrum_uid"] for row in first], [row["spectrum_uid"] for row in second])
        self.assertEqual({group: sum(row["class_original"] == group for row in first) for group in ("young", "aging")}, {"young": 6, "aging": 6})

    def test_separate_group_quotas(self):
        records = []
        for group, label in (("aging", 1), ("young", 0)):
            for subject in ("1", "2"):
                for number in range(10):
                    records.append(
                        {
                            "class_original": group,
                            "class_binary": label,
                            "subject_id": subject,
                            "spectrum_uid": f"{group}-{subject}-{number}",
                        }
                    )
        sample = sample_records_by_group(records, group0_size=7, group1_size=5, seed=31, strategy="balanced")
        self.assertEqual(sum(row["class_binary"] == 0 for row in sample), 7)
        self.assertEqual(sum(row["class_binary"] == 1 for row in sample), 5)
        self.assertEqual({row["subject_id"] for row in sample if row["class_binary"] == 0}, {"1", "2"})

    def test_zero_quota_skips_group(self):
        records = scan_spectra(self.root, blind_order=False)
        sample = sample_records_by_group(records, group0_size=0, group1_size=-1, seed=1)
        self.assertEqual({row["class_binary"] for row in sample}, {1})

    def test_unknown_folder_and_unique_ids(self):
        ordinary = Path(self.temp.name) / "ordinary" / "sample_x"
        ordinary.mkdir(parents=True)
        x = np.linspace(273.14, 3746.91, 1340)
        for number in (1, 2):
            np.savetxt(ordinary / f"Auto-{number}_spec.txt", np.column_stack([x, x + number]), delimiter="\t")
        records = scan_spectra(ordinary, blind_order=False)
        self.assertEqual({row["class_original"] for row in records}, {"unknown"})
        self.assertEqual(len({row["spectrum_uid"] for row in records}), 2)

    def test_sample_manifest_round_trip(self):
        records = scan_spectra(self.root, sample_size=1, seed=12)
        annotation_csv = Path(self.temp.name) / "results" / "annotations.csv"
        save_sample_manifest(records, annotation_csv)
        loaded = load_sample_manifest(annotation_csv)
        self.assertEqual([row["file_absolute_path"] for row in loaded], [row["file_absolute_path"] for row in records])

    def test_read_and_metrics(self):
        record = scan_spectra(self.root, blind_order=False)[0]
        spectrum = read_spectrum(record["file_absolute_path"])
        metrics = calculate_qc_metrics(spectrum)
        self.assertEqual(metrics["n_points"], 1340)
        self.assertEqual(metrics["axis_monotonic"], 1)
        self.assertLess(metrics["intensity_p05"], metrics["intensity_median"])
        self.assertLess(metrics["intensity_median"], metrics["intensity_p95"])
        self.assertGreater(metrics["robust_intensity_range"], 0)
        self.assertGreaterEqual(metrics["finite_fraction"], 1.0)

    def test_pass_rejects_issue(self):
        record = scan_spectra(self.root, blind_order=False)[0]
        spectrum = read_spectrum(record["file_absolute_path"])
        metrics = calculate_qc_metrics(spectrum)
        issues = {field: False for field in ISSUE_FIELDS}
        issues["low_snr"] = True
        with self.assertRaises(ValueError):
            build_annotation(record, metrics, "pass", issues, "")

    def test_issue_fields_match_keyboard_labels_2_to_8(self):
        self.assertEqual(
            ISSUE_FIELDS,
            [
                "vague_finger_peak",
                "missing_finger_peak",
                "low_finger_peak",
                "low_snr",
                "missing_ch_peak",
                "low_ch_peak",
                "cosmic_rays",
            ],
        )


if __name__ == "__main__":
    unittest.main()
