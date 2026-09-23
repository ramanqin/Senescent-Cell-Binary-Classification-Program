"""绘制平均拉曼光谱及与波数轴严格对齐的逐点p值热图。"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np


def _representative_shifts(pointwise, candidate_threshold):
    selected = pointwise[pointwise.p_value < candidate_threshold].sort_values('Raman_shift')
    if selected.empty:
        return []
    shifts = selected.Raman_shift.to_numpy(dtype=float)
    breaks = np.r_[True, np.diff(shifts) > 1.01]
    group_ids = np.cumsum(breaks)
    representatives = []
    for group_id in np.unique(group_ids):
        region = selected.iloc[np.flatnonzero(group_ids == group_id)]
        representatives.append(float(region.loc[region.p_value.idxmin(), 'Raman_shift']))
    return representatives


def plot_mean_spectrum_pvalue_heatmap(matrix, pointwise, destination,
                                      candidate_threshold=.01, display_threshold=.05):
    """上图为全部个体平均谱，下图为逐波数p值色带。"""
    required = {'Raman_shift', 'p_value'}
    if not required.issubset(pointwise.columns):
        raise ValueError('逐点统计表缺少Raman_shift或p_value列')
    if pointwise.Raman_shift.duplicated().any():
        raise ValueError('逐点统计表包含重复波数')
    if not matrix.index.is_unique or not matrix.columns.is_unique:
        raise ValueError('光谱矩阵的波数或个体编号重复')
    if not 0 < candidate_threshold <= display_threshold < 1:
        raise ValueError('阈值必须满足0 < 候选阈值 <= 显示阈值 < 1')

    x = matrix.index.to_numpy(dtype=float)
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(values).all():
        raise ValueError('光谱矩阵包含无效数值')
    if len(x) < 2 or not np.all(np.diff(x) > 0):
        raise ValueError('光谱矩阵波数轴必须严格递增')
    aligned = pointwise.set_index('Raman_shift').reindex(x)
    if aligned.p_value.isna().any():
        raise ValueError('逐点p值与光谱波数轴不完全对应')
    p_values = aligned.p_value.to_numpy(dtype=float)
    if not np.isfinite(p_values).all() or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError('p值必须是0到1之间的有限数值')

    mean_spectrum = values.mean(axis=1)
    positive = p_values[p_values > 0]
    minimum = max(float(positive.min()) if len(positive) else 1e-12, 1e-12)
    displayed = np.clip(p_values, minimum, display_threshold)
    cmap = LinearSegmentedColormap.from_list(
        'raman_pvalue', ['#fff45c', '#ff9d00', '#e31a1c', '#000000'])
    norm = Normalize(vmin=minimum, vmax=display_threshold)

    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False})
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[3.2, 1], hspace=.03)
    spectrum_ax = fig.add_subplot(grid[0])
    heat_ax = fig.add_subplot(grid[1], sharex=spectrum_ax)

    spectrum_ax.plot(x, mean_spectrum, color='black', linewidth=1.6)
    spectrum_ax.set_ylabel('平均拉曼强度（a.u.）')
    spectrum_ax.set_title(f'全部个体平均光谱（n={values.shape[1]}）')
    spectrum_ax.tick_params(axis='x', labelbottom=False)
    representatives = _representative_shifts(pointwise, candidate_threshold)
    span = float(mean_spectrum.max() - mean_spectrum.min()) or 1.
    for index, shift in enumerate(representatives):
        location = int(np.argmin(np.abs(x - shift)))
        y_value = mean_spectrum[location]
        spectrum_ax.axvline(shift, color='#8a8a8a', linewidth=.8, alpha=.75)
        spectrum_ax.annotate(f'{shift:g}', xy=(shift, y_value),
                             xytext=(0, 16 + 12 * (index % 2)), textcoords='offset points',
                             ha='center', va='bottom', rotation=90, fontsize=9,
                             arrowprops={'arrowstyle': '-', 'color': '#777777', 'lw': .8})
    spectrum_ax.set_ylim(mean_spectrum.min() - .06 * span,
                         mean_spectrum.max() + .30 * span)

    step = float(np.median(np.diff(x)))
    image = heat_ax.imshow(displayed[None, :], aspect='auto', interpolation='nearest',
                           extent=[x[0] - step / 2, x[-1] + step / 2, 0, 1],
                           cmap=cmap, norm=norm)
    heat_ax.set_yticks([])
    heat_ax.set_ylabel('逐点p值')
    heat_ax.set_xlabel(r'拉曼位移（cm$^{-1}$）')
    heat_ax.set_xlim(x[0], x[-1])
    colorbar = fig.colorbar(image, ax=[spectrum_ax, heat_ax], fraction=.025, pad=.018)
    ticks = sorted(set([minimum, candidate_threshold, display_threshold]))
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f'{value:.4g}' for value in ticks])
    colorbar.set_label(f'p值（p≥{display_threshold:g}显示为黑色）')
    colorbar.ax.invert_yaxis()
    fig.savefig(destination, dpi=260, facecolor='white')
    plt.close(fig)
    return {'point_count': len(x), 'subject_count': values.shape[1],
            'minimum_p': minimum, 'candidate_threshold': candidate_threshold,
            'display_threshold': display_threshold,
            'representative_shifts': representatives}
