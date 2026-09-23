import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pca_svm_analysis as analysis


class SingleCVTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(13)
        self.X = rng.normal(size=(20, 12))
        self.y = np.repeat([0, 1], 10)
        self.X[:, 0] += 2 * self.y
        self.data = {"matrix": self.X, "labels": self.y, "x": np.arange(12.)}
        self.raw_config = {
            "input_dir": str(self.root), "output_dir": str(self.root / "result"),
            "negative_class": {"name": "年轻", "folders": ["young"]},
            "positive_class": {"name": "衰老", "folders": ["aging"]},
        }
        self.config = analysis.normalize_config(self.raw_config)

    def tearDown(self):
        self.temp.cleanup()

    def test_five_training_fits_plus_one_final_fit_no_inner_search(self):
        original_fit = analysis.Pipeline.fit
        fits = []

        def spy(model, X, y, *args, **kwargs):
            result = original_fit(model, X, y, *args, **kwargs)
            # Both fitted transformers must derive their statistics from this training fold.
            np.testing.assert_allclose(model.named_steps["pca"].mean_, X.mean(axis=0))
            scores = model.named_steps["pca"].transform(X)
            np.testing.assert_allclose(model.named_steps["scale"].mean_, scores.mean(axis=0), atol=1e-12)
            fits.append(X.copy())
            return result

        with patch.object(analysis.Pipeline, "fit", new=spy):
            scores, pred, folds, results = analysis.cross_validation(self.data, self.config)
            final = analysis.fit_final_model(self.data, self.config, self.root)
        self.assertEqual([len(x) for x in fits], [16, 16, 16, 16, 16, 20])
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        for i, (train, test) in enumerate(cv.split(self.X, self.y)):
            np.testing.assert_array_equal(fits[i], self.X[train])
            np.testing.assert_array_equal(folds[test], np.repeat(i+1, len(test)))
        self.assertFalse(np.isnan(scores).any())
        self.assertEqual(len(results), 5)
        self.assertFalse(final["parameter_search_performed"])
        self.assertTrue(all(r["fixed_parameters"]["svm__C"] == .01 for r in results))
        saved = joblib.load(self.root / "model.joblib")
        self.assertEqual(saved["model"].n_features_in_, 12)
        self.assertIn("fixed_parameters", saved)
        self.assertNotIn("inner_selection_score", final)

    def test_minor_class_less_than_five_rejected_without_silent_reduction(self):
        data = {"matrix": self.X[:14], "labels": self.y[:14]}
        with self.assertRaisesRegex(ValueError, "不会自动减少"):
            analysis.cross_validation(data, self.config)

    def test_legacy_config_records_ignored_search_fields(self):
        cfg = analysis.normalize_config({**self.raw_config, "outer_splits": 5, "inner_splits": 4,
                                         "c_values": [100], "gamma_values": [1], "n_jobs": -1})
        self.assertEqual(cfg["cv_splits"], 5)
        self.assertEqual(cfg["svm_c"], .01)
        for k in ("inner_splits", "c_values", "gamma_values", "n_jobs"):
            self.assertNotIn(k, cfg)
            self.assertIn(k, cfg["ignored_legacy_fields"])

    def test_invalid_fixed_parameters_rejected(self):
        for key, value in [("svm_c", 0), ("svm_c", float('nan')), ("svm_gamma", -1),
                           ("svm_gamma", float('inf')), ("svm_kernel", "bad"),
                           ("cv_splits", 1), ("cv_splits", 2.5)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                analysis.normalize_config({**self.raw_config, key: value})

    def test_repeatability(self):
        a = analysis.cross_validation(self.data, self.config)
        b = analysis.cross_validation(self.data, self.config)
        for first, second in zip(a[:3], b[:3]):
            np.testing.assert_array_equal(first, second)

    def test_cli_relative_paths_and_missing_fields(self):
        path = self.root / "config.json"
        path.write_text(json.dumps({**self.raw_config, "input_dir": ".", "output_dir": "out"}), encoding='utf-8')
        cfg = analysis.load_config(path)
        self.assertEqual(Path(cfg["input_dir"]), self.root)
        self.assertEqual(Path(cfg["output_dir"]), self.root / "out")
        path.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '配置缺少'):
            analysis.load_config(path)


if __name__ == '__main__':
    unittest.main()
