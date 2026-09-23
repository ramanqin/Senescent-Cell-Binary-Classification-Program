"""验证差异分析入口只接收已预处理光谱，并按个体平均。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from compact_result import export_compact
from run_analysis import EXPECTED_X, _read_spectrum, run


class PreprocessedInputTests(unittest.TestCase):
    def _make_input(self, root, x=EXPECTED_X):
        for group, offset in (('年轻', 0.0), ('衰老', 0.02)):
            for number in range(2):
                folder = root / group / f'个体{number + 1}'
                folder.mkdir(parents=True)
                for repeat in range(2):
                    y = (0.03 + 0.02 * np.exp(-((x - 1000) / 35) ** 2)
                         + offset + number * 0.001 + repeat * 0.002)
                    np.savetxt(folder / f'光谱{repeat + 1}.txt', np.column_stack((x, y)))

    def test_individual_mean_and_compact_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root, work, out = root / 'input', root / 'work', root / 'out'
            self._make_input(input_root)
            out.mkdir()
            run(input_root, work)
            export_compact(work, out)
            matrix = pd.read_csv(work / '个体平均光谱矩阵.csv', index_col=0)
            self.assertEqual(matrix.shape, (1201, 4))
            self.assertAlmostEqual(matrix.loc[1000.0, '年轻/个体1'], 0.051)
            summary = json.loads((work / '运行摘要.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['processed_spectra'], 8)
            self.assertTrue((out / '结果说明.txt').is_file())
            self.assertTrue((out / '结果图.png').is_file())
            self.assertNotIn('清洗保留', (out / '结果说明.txt').read_text(encoding='utf-8-sig'))

    def test_wrong_grid_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_input(root / 'input', x=np.arange(600.0, 1801.0, 3.0))
            with self.assertRaisesRegex(ValueError, '1201 点'):
                run(root / 'input', root / 'work')

    def test_batch_output_header_in_txt_and_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            intensity = np.linspace(0.01, 0.03, len(EXPECTED_X))
            for suffix, delimiter in (('.txt', '\t'), ('.csv', ',')):
                path = root / f'batch{suffix}'
                np.savetxt(path, np.column_stack((EXPECTED_X, intensity)),
                           delimiter=delimiter, fmt='%.8g',
                           header=f'Raman_shift_cm-1{delimiter}Intensity', comments='')
                np.testing.assert_allclose(_read_spectrum(path), intensity, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
