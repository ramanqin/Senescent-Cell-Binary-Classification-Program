"""改编自用户提供的004_Ecoli_spectrum_violin_plot-main/violin_plot.py。

复用collect_value的窗口梯形积分、draw_violin的小提琴叠加窄箱线图逻辑。
适配：三组改为两组；TXT输入改为个体平均谱矩阵；不删除面积极端个体；
已完成预处理，不重复基线校正。沿用既有候选窗口，不运行004的共识峰筛选。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.stats import mannwhitneyu


def collect_value(band_range, x, subject_spectra):
    """004面积计算的矩阵输入版：一列代表一个个体，不是一次采集。"""
    start_x, end_x = band_range
    mask = (x >= start_x) & (x <= end_x)
    if mask.sum() < 2 or start_x < x.min() or end_x > x.max():
        raise ValueError(f'积分窗口{start_x:g}–{end_x:g}没有被光谱完整覆盖')
    # 对应004 collect_value(baseline_correct=False, outlier=False)。
    # 已完成ALS校正，不再做端点基线或负值截零；保留全部质控合格个体。
    return trapezoid(subject_spectra[mask], x[mask], axis=0)


def draw_violin(plot_data_list, ax, rng):
    """复用004：隐藏内置极值线，在小提琴内叠加黑色窄箱线图。"""
    colors = ['#929292', '#FF599B']
    for pos, area, color in zip([1, 2], plot_data_list, colors):
        # 常数数据无法估计核密度，但仍可显示箱线图及个体散点。
        if np.ptp(area) > 0:
            violin = ax.violinplot(area, positions=[pos], widths=.72,
                                  showmeans=False, showmedians=False, showextrema=False)
            violin['bodies'][0].set(facecolor=color, edgecolor='#0000007A', alpha=.3)
        ax.scatter(pos+rng.uniform(-.11, .11, len(area)), area,
                   s=20, alpha=.75, color=color, edgecolors='white', linewidths=.4, zorder=2)
    # showfliers=False仅关闭箱线图重复绘制的离群点；上面的散点仍展示所有个体。
    ax.boxplot(plot_data_list, positions=[1, 2], widths=.12, showfliers=False,
        whis=1.5, boxprops=dict(color='black', linewidth=.8),
        whiskerprops=dict(color='black', linewidth=.8),
        capprops=dict(color='black', linewidth=.8),
        medianprops=dict(color='black', linewidth=1.1), zorder=4)


def plot_peak_violin(matrix, regions, destination, half_width=8):
    """在每个候选区间最小p值处取±8窗口；它未必是局部峰顶。"""
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 11})
    x = matrix.index.to_numpy(dtype=float)
    if not np.all(np.diff(x) > 0):
        raise ValueError('波数轴必须递增')
    arrays = [matrix.loc[:, matrix.columns.str.startswith(g+'/')].to_numpy()
              for g in ['年轻', '衰老']]
    if min(a.shape[1] for a in arrays) < 2:
        raise ValueError('每组至少需要两个个体')
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('光谱存在非有限值')
    count = len(regions)
    rows = max(1, int(np.ceil((count+1)/3)))
    fig, axes = plt.subplots(rows, 3, figsize=(14, 4.4*rows), squeeze=False)
    axes = axes.ravel()
    rng = np.random.default_rng(20260922)
    report = []
    for ax, region in zip(axes, regions.itertuples(index=False)):
        center = float(region.representative_shift)
        lo, hi = center-half_width, center+half_width
        areas = [collect_value((lo, hi), x, a) for a in arrays]
        p = float(mannwhitneyu(*areas, alternative='two-sided').pvalue)
        draw_violin(areas, ax, rng)
        ax.set_xticks([1, 2], [f'健康（年轻）\nn={len(areas[0])}', f'衰老\nn={len(areas[1])}'])
        ax.set_xlim(.45, 2.55)
        ax.set_ylabel(r'归一化光谱积分面积（a.u.·cm$^{-1}$）')
        ax.set_title(f'{center:g} '+r'cm$^{-1}$'+f'（{lo:g}–{hi:g}）\n峰面积组间比较：p={p:.4f}', fontsize=12)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        report.append(dict(center=center, window_start=lo, window_end=hi,
            young_n=len(areas[0]), aging_n=len(areas[1]), raw_area_p=p,
            young_mean_area=float(areas[0].mean()), aging_mean_area=float(areas[1].mean())))
    for ax in axes[count:]:
        ax.axis('off')
    fig.suptitle('健康（年轻）与衰老组：候选波段面积分布', fontsize=18, y=.99)
    fig.tight_layout(rect=[0, 0, 1, .95], h_pad=2.4, w_pad=2)
    fig.savefig(destination, dpi=300, facecolor='white')
    plt.close(fig)
    return report
