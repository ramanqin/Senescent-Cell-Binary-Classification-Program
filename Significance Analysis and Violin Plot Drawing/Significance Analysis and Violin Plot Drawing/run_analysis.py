"""读取 data analysis 已预处理的光谱，按个体平均后做差异分析。"""

import argparse
import hashlib
import json
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from compact_result import export_compact
from differential_analysis import FDR_ALPHA, analyze


PREPROCESSED_DATA_DIR = None  # 也可用 --input 指定 preprocessing_run_* 目录
RAW_P_THRESHOLD = 0.01
VIOLINS_PER_PAGE = 10
INCLUDE_AREA_VIOLIN = False
EXPECTED_X = np.arange(600.0, 1801.0, 1.0)
GROUPS = ('年轻', '衰老')


def _read_spectrum(path):
    delimiter = ',' if path.suffix.lower() == '.csv' else None
    try:
        with path.open('r', encoding='utf-8-sig') as stream:
            first_line = stream.readline().strip().lower().replace(' ', '')
        # Spectral Batch Processing exports this one-line header by default.
        batch_headers = {'raman_shift_cm-1\tintensity', 'raman_shift_cm-1,intensity'}
        data = np.loadtxt(path, delimiter=delimiter, encoding='utf-8-sig',
                          skiprows=1 if first_line in batch_headers else 0)
    except (OSError, ValueError) as exc:
        raise ValueError(f'无法读取两列数值光谱：{path}（{exc}）') from exc
    if data.shape != (len(EXPECTED_X), 2):
        raise ValueError(f'不是 600–1800 cm⁻¹、步长 1 的 1201 点双列光谱：{path}；实际形状 {data.shape}')
    if not np.isfinite(data).all():
        raise ValueError(f'光谱包含非有限数值：{path}')
    if not np.allclose(data[:, 0], EXPECTED_X, rtol=0, atol=1e-6):
        raise ValueError(f'波数轴不符合 600–1800 cm⁻¹、步长 1：{path}')
    return data[:, 1]


def _digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(input_root, work, raw_p_threshold=RAW_P_THRESHOLD):
    """生成临时统计文件；调用方再用 export_compact 导出简明结果。"""
    input_root = Path(input_root).resolve()
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    if not input_root.is_dir():
        raise ValueError(f'输入目录不存在：{input_root}')

    spectra = []
    labels = []
    subject_ids = []
    counts = {}
    inventory = []
    for group in GROUPS:
        group_dir = input_root / group
        if not group_dir.is_dir():
            raise ValueError(f'缺少组目录：{group_dir}')
        subjects = sorted(p for p in group_dir.iterdir() if p.is_dir())
        if len(subjects) < 2:
            raise ValueError(f'{group}组至少需要 2 个独立个体，当前 {len(subjects)} 个')
        counts[group] = len(subjects)
        for subject in subjects:
            files = sorted(p for p in subject.iterdir()
                           if p.is_file() and p.suffix.lower() in ('.txt', '.csv'))
            if not files:
                raise ValueError(f'个体目录没有 TXT/CSV 光谱：{subject}')
            values = []
            for path in files:
                values.append(_read_spectrum(path))
                inventory.append({'relative_path': str(path.relative_to(input_root)),
                                  'sha256': _digest(path)})
            spectra.append(np.mean(values, axis=0))
            labels.append(group)
            subject_ids.append(f'{group}/{subject.name}')

    matrix = pd.DataFrame(np.column_stack(spectra), index=EXPECTED_X, columns=subject_ids)
    matrix.index.name = 'Raman_shift'
    matrix.to_csv(work / '个体平均光谱矩阵.csv', encoding='utf-8-sig')
    stats = analyze(EXPECTED_X, np.vstack(spectra), np.asarray(labels),
                    work / '差异性结果', raw_p_threshold)
    for entry in inventory:
        path = input_root / entry['relative_path']
        if not path.is_file() or _digest(path) != entry['sha256']:
            raise RuntimeError(f'分析期间输入光谱发生变化：{path}')
    parameters = {
        'input_directory': str(input_root),
        'input_stage': 'data analysis 已清洗和预处理的光谱',
        'expected_range_cm_minus_1': [600, 1800],
        'expected_step_cm_minus_1': 1,
        'raw_p_threshold': raw_p_threshold,
        'fdr_alpha': FDR_ALPHA,
        'processed_spectra': len(inventory),
        'subject_counts': counts,
    }
    (work / '实际使用参数.json').write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding='utf-8')
    (work / '运行摘要.json').write_text(json.dumps({
        'summary': stats, 'processed_spectra': len(inventory), 'subject_counts': counts,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description='对已清洗、预处理的年轻/衰老光谱做差异分析')
    parser.add_argument('--input', type=Path, default=PREPROCESSED_DATA_DIR,
                        help='data analysis 输出的 preprocessing_run_* 目录')
    parser.add_argument('--output-root', type=Path, default=Path(__file__).parent / 'result',
                        help='结果根目录；每次新建一个时间戳子目录')
    parser.add_argument('--raw-p-threshold', type=float, default=RAW_P_THRESHOLD)
    parser.add_argument('--violins-per-page', type=int, default=VIOLINS_PER_PAGE)
    parser.add_argument('--include-area', action='store_true', default=INCLUDE_AREA_VIOLIN)
    args = parser.parse_args(argv)
    if args.input is None:
        parser.error('请用 --input 指定预处理输出目录，或设置 PREPROCESSED_DATA_DIR')
    if not 0 < args.raw_p_threshold < 1:
        parser.error('--raw-p-threshold 必须在 0 和 1 之间')
    if args.violins_per_page < 1:
        parser.error('--violins-per-page 必须大于 0')

    output_root = args.output_root.resolve()
    input_root = args.input.resolve()
    if output_root == input_root or input_root in output_root.parents:
        parser.error('输出目录不能放在输入光谱目录内')
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / f'analysis_run_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir()
    try:
        with tempfile.TemporaryDirectory(prefix='raman_significance_') as temporary:
            work = Path(temporary)
            run(input_root, work, args.raw_p_threshold)
            export_compact(work, out, args.violins_per_page, args.include_area)
    except Exception:
        (out / '运行失败.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    print(f'结果目录：{out}')
    return out


if __name__ == '__main__':
    main()
