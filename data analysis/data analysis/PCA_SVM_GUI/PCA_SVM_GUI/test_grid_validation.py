import tempfile
import unittest
from pathlib import Path

import numpy as np

from pca_svm_analysis import load_samples, run_analysis


class GridValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.x = np.arange(600.0, 1801.0, 3.0)
        for class_name in ("young", "aging"):
            for subject in ("01", "02"):
                folder = self.root / class_name / subject
                folder.mkdir(parents=True)
                y = np.sin(self.x / 100.0) + (class_name == "aging") * 0.1
                np.savetxt(folder / "spectrum.txt", np.column_stack([self.x, y]), delimiter="\t")

    def tearDown(self):
        self.temporary.cleanup()

    def config(self):
        return {
            "input_dir": str(self.root),
            "output_dir": str(self.root / "result_plot"),
            "negative_class": {"name": "年轻", "folders": ["young"]},
            "positive_class": {"name": "衰老", "folders": ["aging"]},
            "extensions": ["txt"],
            "sample_depth_after_class": 1,
            "grid_tolerance": 1e-6,
            "min_points": 20,
        }

    def test_equal_grids_are_accepted_without_resampling(self):
        data = load_samples(self.config())
        self.assertTrue(data["grid_info"]["validated_equal"])
        np.testing.assert_array_equal(data["x"], self.x)

    def test_mismatched_grid_is_rejected(self):
        path = self.root / "aging" / "02" / "spectrum.txt"
        y = np.sin(self.x / 100.0) + 0.1
        np.savetxt(path, np.column_stack([self.x + 0.1, y]), delimiter="\t")
        with self.assertRaisesRegex(ValueError, "不会自动插值"):
            load_samples(self.config())

    def test_same_subject_folder_in_different_batches_is_not_merged(self):
        for batch in ("P4", "P7"):
            for class_name in ("young", "aging"):
                folder = self.root / batch / class_name / "01"
                folder.mkdir(parents=True)
                y = np.sin(self.x / 100.0) + (class_name == "aging") * 0.1
                np.savetxt(folder / "spectrum.txt", np.column_stack([self.x, y]), delimiter="\t")
        data = load_samples(self.config())
        p4 = [sample for sample in data["sample_ids"] if sample.startswith("P4/")]
        p7 = [sample for sample in data["sample_ids"] if sample.startswith("P7/")]
        self.assertEqual(len(p4), 2)
        self.assertEqual(len(p7), 2)
        self.assertEqual(len(data["sample_ids"]), 8)

    def test_analysis_creates_versioned_output_and_input_manifest(self):
        for class_name in ("young", "aging"):
            for subject in ("03", "04"):
                folder = self.root / class_name / subject
                folder.mkdir(parents=True)
                y = np.sin(self.x / 100.0) + (class_name == "aging") * 0.25 + int(subject) * 0.001
                np.savetxt(folder / "spectrum.txt", np.column_stack([self.x, y]), delimiter="\t")
        config = self.config()
        config.update({
            "outer_splits": 2,
            "inner_splits": 2,
            "random_seed": 42,
            "pca_variance": 0.95,
            "class_weight": "balanced",
            "n_jobs": 1,
            "c_values": [1.0],
            "gamma_values": ["scale"],
            "roc_filename": "ROC.png",
        })
        result = run_analysis(config)
        output = Path(result["output_dir"])
        self.assertEqual(output.parent, Path(config["output_dir"]))
        self.assertTrue(output.name.startswith("analysis_run_"))
        self.assertTrue((output / "run_manifest.csv").is_file())
        manifest_lines = (output / "run_manifest.csv").read_text(encoding="utf-8-sig").splitlines()
        self.assertEqual(len(manifest_lines), 9)


if __name__ == "__main__":
    unittest.main()
