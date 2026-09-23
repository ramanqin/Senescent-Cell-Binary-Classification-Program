import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from intensity_violin import select_points, export_selected_violins, plot_intensity_violin


class LinkedViolinTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_threshold_is_dynamic_and_strict(self):
        table = pd.DataFrame({'Raman_shift':[635,636,637], 'p_value':[.009,.01,.04], 'q_value':[.2]*3})
        self.assertEqual(select_points(table,.01).Raman_shift.tolist(), [635])
        self.assertEqual(select_points(table,.05).Raman_shift.tolist(), [635,636,637])

    def test_invalid_threshold(self):
        for value in [0,1,-1,float('nan')]:
            with self.assertRaises(ValueError):
                select_points(pd.DataFrame(), value)

    def test_duplicate_points_rejected(self):
        with self.assertRaises(ValueError):
            select_points(pd.DataFrame({'Raman_shift':[600,600], 'p_value':[.01,.02], 'q_value':[.1,.1]}), .05)

    def test_invalid_p_rejected(self):
        with self.assertRaises(ValueError):
            select_points(pd.DataFrame({'Raman_shift':[600], 'p_value':[-.1], 'q_value':[.1]}), .05)

    def test_pagination_keeps_every_point(self):
        for count in [1,10,11,67]:
            selected = pd.DataFrame({'Raman_shift':np.arange(600,600+count), 'p_value':[.005]*count})
            def fake_plot(matrix, shifts, destination, expected):
                return [{'wavenumber':s} for s in shifts]
            with TemporaryDirectory() as tmp, patch('intensity_violin.plot_intensity_violin', side_effect=fake_plot):
                names, report = export_selected_violins(pd.DataFrame(), selected, Path(tmp), 10)
                self.assertEqual(len(names), (count+9)//10)
                self.assertEqual([r['wavenumber'] for r in report], selected.Raman_shift.tolist())

    def test_no_candidates(self):
        with TemporaryDirectory() as tmp:
            names, report = export_selected_violins(pd.DataFrame(), pd.DataFrame(columns=['Raman_shift','p_value']), Path(tmp))
            self.assertEqual(report, [])
            self.assertTrue((Path(tmp)/names[0]).is_file())

    def test_constant_intensity(self):
        matrix = pd.DataFrame([[1.,1.,1.,1.]], index=[600], columns=['年轻/1','年轻/2','衰老/1','衰老/2'])
        with TemporaryDirectory() as tmp:
            report = plot_intensity_violin(matrix,[600],Path(tmp)/'test.png',{600:1.})
            self.assertEqual(report[0]['p_value'],1.)

    def test_p_mismatch_stops(self):
        matrix = pd.DataFrame([[1.,1.,1.,1.]], index=[600], columns=['年轻/1','年轻/2','衰老/1','衰老/2'])
        with TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError,'不一致'):
            plot_intensity_violin(matrix,[600],Path(tmp)/'bad.png',{600:.001})

    def test_missing_wavenumber_stops(self):
        matrix = pd.DataFrame([[1.,1.,1.,1.]], index=[600], columns=['年轻/1','年轻/2','衰老/1','衰老/2'])
        with TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError,'不存在'):
            plot_intensity_violin(matrix,[601],Path(tmp)/'bad.png')


if __name__ == '__main__':
    unittest.main()
