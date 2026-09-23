"""直接接收本轮差异分析结果，自动绘制候选点强度图，无历史数据路径。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from peak_violin import draw_violin

def select_points(point_table, threshold):
    """严格使用原始p<阈值，不合并相邻点，不固定波数位置。"""
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('筛选阈值必须在0与1之间')
    if not {'Raman_shift', 'p_value', 'q_value'}.issubset(point_table.columns):
        raise ValueError('差异分析表缺少波数、p值或q值列')
    if point_table.Raman_shift.duplicated().any():
        raise ValueError('差异分析表包含重复波数')
    if not np.isfinite(point_table[['Raman_shift', 'p_value', 'q_value']].to_numpy()).all():
        raise ValueError('差异分析表包含无效数值')
    if not point_table[['p_value','q_value']].apply(lambda col: col.between(0, 1).all()).all():
        raise ValueError('p值或q值超出0–1范围')
    return point_table.loc[point_table.p_value < threshold].sort_values('Raman_shift').copy()


def plot_intensity_violin(matrix, wavenumbers, destination, expected_p=None):
    if not wavenumbers or len(set(wavenumbers)) != len(wavenumbers):
        raise ValueError('波数列表不能为空或重复')
    if not matrix.index.is_unique:
        raise ValueError('波数轴不能有重复值')
    if not matrix.columns.is_unique:
        raise ValueError('个体编号不能重复')
    group_masks = [matrix.columns.str.startswith(g+'/') for g in ['年轻', '衰老']]
    if min(mask.sum() for mask in group_masks) < 2:
        raise ValueError('每组至少需要2个个体')
    if not np.all(group_masks[0] | group_masks[1]):
        raise ValueError('矩阵列名必须以年轻/或衰老/开头')
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 11})
    columns = min(5, len(wavenumbers))
    rows = int(np.ceil(len(wavenumbers)/columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4*columns, 4.8*rows), squeeze=False)
    rng = np.random.default_rng(20260922)
    report = []
    for ax, shift in zip(axes.ravel(), sorted(wavenumbers)):
        if shift not in matrix.index:
            raise ValueError(f'矩阵中不存在精确波数{shift}，未自动插值或取邻点')
        values = [matrix.loc[shift, mask].to_numpy(dtype=float) for mask in group_masks]
        if not all(np.isfinite(v).all() for v in values):
            raise ValueError(f'{shift}处有无效强度值')
        p = float(mannwhitneyu(*values, alternative='two-sided').pvalue)
        if expected_p is not None:
            # 图和差异分析必须用同一批个体、同一强度及同一统计方法。
            if shift not in expected_p or not np.isclose(p, expected_p[shift], rtol=1e-10, atol=1e-14):
                raise ValueError(f'{shift:g}处小提琴检验p值与差异分析不一致，停止输出')
        draw_violin(values, ax, rng)
        ax.set_xticks([1, 2], [f'健康（年轻）\nn={len(values[0])}', f'衰老\nn={len(values[1])}'])
        ax.set_xlim(.45, 2.55)
        ax.set_ylabel('归一化光谱强度（a.u.）')
        p_text = f'{p:.4f}' if p >= .0001 else f'{p:.2e}'
        ax.set_title(f'{shift:g} '+r'cm$^{-1}$'+f'\n强度组间比较：p={p_text}', fontsize=12)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        report.append({'wavenumber': shift, 'p_value': p,
            'young_n': len(values[0]), 'aging_n': len(values[1])})
    for ax in axes.ravel()[len(wavenumbers):]:
        ax.axis('off')
    fig.suptitle(f'健康（年轻）与衰老组：{len(wavenumbers)}个候选波数点的强度分布', fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, .95], h_pad=2.5, w_pad=1.8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300, facecolor='white')
    plt.close(fig)
    return report


def export_selected_violins(matrix, selected, out, per_page=10):
    """绘制筛选表全部入选点；没有入选点时生成说明图。"""
    if isinstance(per_page, bool) or not isinstance(per_page, int) or not 1 <= per_page <= 20:
        raise ValueError('每页数量必须是1–20之间的整数')
    selected = selected.sort_values('Raman_shift')
    if selected.Raman_shift.duplicated().any():
        raise ValueError('候选波数不能重复')
    out.mkdir(parents=True, exist_ok=True)
    if selected.empty:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis('off')
        ax.text(.5, .5, '本轮没有符合筛选条件的波数点\n未绘制强度小提琴图',
                ha='center', va='center', fontsize=16)
        filename = out/'强度小提琴图.png'
        fig.savefig(filename, dpi=180, bbox_inches='tight')
        plt.close(fig)
        return [filename.name], []
    shifts = selected.Raman_shift.tolist()
    expected = dict(zip(shifts, selected.p_value))
    pages = (len(shifts)+per_page-1)//per_page
    names, reports = [], []
    for page in range(pages):
        suffix = '' if pages == 1 else f'_{page+1:02d}'
        filename = out/f'强度小提琴图{suffix}.png'
        reports.extend(plot_intensity_violin(matrix, shifts[page*per_page:(page+1)*per_page], filename, expected))
        names.append(filename.name)
    if len(reports) != len(selected):
        raise RuntimeError('绘图点数与筛选点数不一致')
    return names, reports


if __name__ == '__main__':
    raise SystemExit('请运行run_analysis.py；本模块自动接收本次差异分析结果。')
