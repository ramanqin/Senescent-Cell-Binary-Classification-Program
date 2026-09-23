"""将本轮候选波数点绘制为个体层面的强度热图。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def plot_intensity_heatmap(matrix, selected, destination):
    """行是候选波数点，列是个体；颜色为每个位点内跨个体Z-score。"""
    required = {'Raman_shift', 'p_value'}
    if not required.issubset(selected.columns):
        raise ValueError('候选表缺少Raman_shift或p_value列')
    if selected.Raman_shift.duplicated().any():
        raise ValueError('候选表包含重复波数')
    if not matrix.index.is_unique or not matrix.columns.is_unique:
        raise ValueError('光谱矩阵的波数或个体编号重复')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False})
    selected = selected.sort_values('Raman_shift')
    if selected.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis('off')
        ax.text(.5, .5, '本轮没有符合筛选条件的波数点\n未绘制候选位点热图',
                ha='center', va='center', fontsize=16)
        fig.savefig(destination, dpi=180, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return {'point_count': 0, 'subject_count': 0, 'constant_points': []}

    shifts = selected.Raman_shift.to_numpy(dtype=float)
    missing = [value for value in shifts if value not in matrix.index]
    if missing:
        raise ValueError('光谱矩阵缺少候选波数：' + '、'.join(f'{value:g}' for value in missing))
    young = sorted(column for column in matrix.columns if column.startswith('年轻/'))
    aging = sorted(column for column in matrix.columns if column.startswith('衰老/'))
    if min(len(young), len(aging)) < 2 or len(young) + len(aging) != len(matrix.columns):
        raise ValueError('矩阵列名必须全部以年轻/或衰老/开头，且每组至少2个个体')
    subjects = young + aging
    values = matrix.loc[shifts, subjects].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('候选位点强度包含无效数值')

    means = values.mean(axis=1, keepdims=True)
    stds = values.std(axis=1, ddof=1, keepdims=True)
    constant = shifts[np.squeeze(stds <= np.finfo(float).eps)].tolist()
    safe_stds = np.where(stds > np.finfo(float).eps, stds, 1.)
    z = (values - means) / safe_stds
    z[stds[:, 0] <= np.finfo(float).eps] = 0.

    width = max(13, min(24, .30 * len(subjects) + 5))
    height = max(5.8, min(14, .55 * len(shifts) + 3.2))
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[.28, 5], hspace=.03)
    group_ax = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1], sharex=group_ax)

    group_codes = np.array([[0] * len(young) + [1] * len(aging)])
    group_colors = ['#929292', '#FF599B']
    group_ax.imshow(group_codes, aspect='auto', interpolation='nearest',
                    cmap=ListedColormap(group_colors), vmin=0, vmax=1)
    group_ax.set_yticks([])
    group_ax.tick_params(axis='x', bottom=False, labelbottom=False)
    for spine in group_ax.spines.values():
        spine.set_visible(False)
    group_ax.legend(handles=[Patch(facecolor=group_colors[0], label=f'健康（年轻），n={len(young)}'),
                             Patch(facecolor=group_colors[1], label=f'衰老，n={len(aging)}')],
                    loc='lower center', bbox_to_anchor=(.5, 1.03), ncol=2, frameon=False)

    image = ax.imshow(z, aspect='auto', interpolation='nearest', cmap='RdBu_r', vmin=-2.5, vmax=2.5)
    ax.set_yticks(np.arange(len(shifts)), [f'{value:g}' for value in shifts])
    ax.set_ylabel(r'候选波数点（cm$^{-1}$）')
    ax.set_xticks(np.arange(len(subjects)), [subject.split('/', 1)[1] for subject in subjects],
                  rotation=90, fontsize=7)
    ax.set_xlabel('个体编号')
    boundary = len(young) - .5
    group_ax.axvline(boundary, color='white', linewidth=2)
    ax.axvline(boundary, color='black', linewidth=1.2)
    colorbar = fig.colorbar(image, ax=ax, fraction=.025, pad=.015)
    colorbar.set_label('位点内Z-score（显示范围 −2.5 至 2.5）')
    fig.suptitle(f'候选波数点强度热图（{len(shifts)}个位点，{len(subjects)}个个体）', fontsize=16)
    fig.savefig(destination, dpi=260, facecolor='white')
    plt.close(fig)
    return {'point_count': len(shifts), 'subject_count': len(subjects),
            'young_subjects': len(young), 'aging_subjects': len(aging),
            'constant_points': constant}
