"""逐波数及候选峰面积检验；由主程序传入本次个体平均光谱。"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from statsmodels.stats.multitest import multipletests

RAW_P_THRESHOLD = 0.01  # 原始p值的探索性筛选阈值
FDR_ALPHA = 0.05       # 保留原FDR口径，和原始p阈值分开
PEAK_PROMINENCE_RATIO = 0.05  # 找峰参数，不是显著性阈值
PEAK_DISTANCE = 8
PEAK_HALF_WIDTH = 8


def analyze(x, X, labels, out, raw_p_threshold=0.01):
    """接收已预处理光谱的个体平均谱，不依赖历史结果或内置数据。"""
    RAW_P_THRESHOLD = raw_p_threshold
    if not 0 < RAW_P_THRESHOLD < 1:
        raise ValueError('原始p值阈值必须在0到1之间')
    out.mkdir(parents=True, exist_ok=False)
    young, old = X[labels == '年轻'], X[labels == '衰老']
    if min(len(young), len(old)) < 2:
        raise ValueError('每组至少需要2个有合格光谱的个体')
    if X.shape != (len(labels), len(x)) or not np.isfinite(X).all():
        raise ValueError('个体平均谱形状或数值异常')
    # 和原statistics函数一致：每个波数点进行双侧Mann–Whitney U检验。
    p = np.array([mannwhitneyu(a, b, alternative='two-sided').pvalue
                  for a, b in zip(young.T, old.T)])
    reject, q, _, _ = multipletests(p, alpha=FDR_ALPHA, method='fdr_bh')
    point = pd.DataFrame(dict(Raman_shift=x, young_mean=young.mean(0),
        old_mean=old.mean(0), difference=young.mean(0)-old.mean(0),
        p_value=p, q_value=q, raw_p_selected=p < RAW_P_THRESHOLD,
        fdr_significant=reject))
    point['direction'] = np.where(point.difference > 0, '年轻较高', '衰老较高')
    selected = point[point.raw_p_selected].copy()
    regions = []
    indices = np.flatnonzero(p < RAW_P_THRESHOLD)
    for group in np.split(indices, np.where(np.diff(indices) > 1)[0]+1):
        if not len(group):
            continue
        part = point.iloc[group]
        representative = part.loc[part.p_value.idxmin()]
        regions.append(dict(start=x[group[0]], end=x[group[-1]], point_count=len(group),
            representative_shift=representative.Raman_shift, min_p=representative.p_value,
            q_at_representative=representative.q_value,
            direction=' / '.join(part.direction.unique())))
    region_df = pd.DataFrame(regions, columns=['start','end','point_count','representative_shift','min_p','q_at_representative','direction'])
    # 峰面积检验也沿用原参数，单独统计，不把波数点与峰混为一谈。
    peaks, _ = find_peaks(X.mean(0), prominence=PEAK_PROMINENCE_RATIO*np.ptp(X.mean(0)),
                         distance=int(PEAK_DISTANCE / np.diff(x)[0]))
    peak_rows = []
    for i in peaks:
        mask = (x >= x[i]-PEAK_HALF_WIDTH) & (x <= x[i]+PEAK_HALF_WIDTH)
        areas = trapezoid(X[:, mask], x[mask], axis=1)
        a, b = areas[labels == '年轻'], areas[labels == '衰老']
        u, pv = mannwhitneyu(a, b, alternative='two-sided')
        peak_rows.append(dict(peak_shift=x[i], young_mean_area=a.mean(),
            old_mean_area=b.mean(), p_value=pv, effect_size=2*u/(len(a)*len(b))-1))
    peak_df = pd.DataFrame(peak_rows, columns=['peak_shift','young_mean_area','old_mean_area','p_value','effect_size'])
    peak_df['q_value'] = multipletests(peak_df.p_value, alpha=FDR_ALPHA, method='fdr_bh')[1] if len(peak_df) else pd.Series(dtype=float)
    peak_df['raw_p_selected'] = peak_df.p_value < RAW_P_THRESHOLD
    prefix = f'原始p小于{RAW_P_THRESHOLD:g}'
    for name, table in [('全部波数点_保留p和q', point), (prefix+'_波数点', selected),
                        (prefix+'_连续区间', region_df), ('全部候选峰面积', peak_df),
                        (prefix+'_候选峰面积', peak_df[peak_df.raw_p_selected])]:
        table.to_csv(out / (name+'.csv'), index=False, encoding='utf-8-sig')
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(12, 4.6), constrained_layout=True)
    ax.plot(x, -np.log10(np.clip(p, 1e-300, 1)), color='#2474a7', lw=1.2, label='原始p值')
    for threshold, color in [(0.05, '#999999'), (RAW_P_THRESHOLD, '#d56c43')]:
        ax.axhline(-np.log10(threshold), ls='--', color=color, label=f'p={threshold:g}')
    ax.scatter(selected.Raman_shift, -np.log10(np.clip(selected.p_value, 1e-300, 1)), color='#d56c43', s=28,
               label=f'p<{RAW_P_THRESHOLD:g}：{len(selected)}个波数点', zorder=3)
    ax.set(xlabel=r'拉曼位移（cm$^{-1}$）', ylabel=r'$-\log_{10}(p)$',
           title=f'原始p值探索性筛选：0.05阈值{int((p<.05).sum())}点，{RAW_P_THRESHOLD:g}阈值{len(selected)}点')
    ax.legend(loc='upper center', ncol=2)
    fig.savefig(out/'原始p值阈值对照.png', dpi=220)
    fig.savefig(out/'原始p值阈值对照.svg')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4.6), constrained_layout=True)
    for name, values, color in [('年轻', young, '#2474a7'), ('衰老', old, '#d56c43')]:
        mean = values.mean(0)
        sem = values.std(0, ddof=1) / np.sqrt(len(values))
        ax.plot(x, mean, color=color, label=f'{name}（n={len(values)}）')
        ax.fill_between(x, mean-sem, mean+sem, color=color, alpha=.15)
    ax.set(xlabel=r'拉曼位移（cm$^{-1}$）', ylabel='向量归一化强度', title='预处理光谱的个体平均：组均值 ± SEM')
    ax.legend()
    fig.savefig(out/'两组平均光谱.png', dpi=220)
    fig.savefig(out/'两组平均光谱.svg')
    plt.close(fig)
    summary = dict(raw_p_threshold=RAW_P_THRESHOLD, fdr_alpha=FDR_ALPHA,
        young_subjects=len(young), aging_subjects=len(old), point_count=len(x),
        original_p005_count=int((p < .05).sum()), selected_point_count=len(selected),
        retained_wavenumbers=selected.Raman_shift.tolist(), interval_count=len(regions),
        candidate_peak_count=len(peak_df), selected_peak_count=int(peak_df.raw_p_selected.sum()),
        fdr_significant_points=int(reject.sum()))
    (out/'筛选摘要.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = [f'# 原始p<{RAW_P_THRESHOLD:g}探索性筛选', '',
        f'本次输入的已预处理光谱中，{len(young)}个年轻个体、{len(old)}个衰老个体参与统计。{len(x)}个波数点中，p<0.05共{int((p<.05).sum())}点，p<{RAW_P_THRESHOLD:g}有{len(selected)}点。',
        '', '波数点：'+ '、'.join(str(int(v)) for v in selected.Raman_shift)+' cm⁻¹。', '',
        f'合并相邻点得到{len(regions)}个连续区间。该合并仅用于整理候选位置，不是区间层面的显著性检验。', '',
        f'{len(peak_df)}个候选峰的积分面积检验中，原始p<{RAW_P_THRESHOLD:g}有{int(peak_df.raw_p_selected.sum())}个。波数点与峰面积是两类检验。', '',
        f'保留BH-FDR校正值及q<0.05判断，FDR通过点为{int(reject.sum())}。原始p候选点用于探索性峰归属，不代表已确认差异或因果因素。', '',
        '各个体使用合格光谱算术平均，统计时等权。个体编号不一定等同于独立供体或培养批次，独立性需结合实验记录确认。']
    (out/'结果说明.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


if __name__ == '__main__':
    raise SystemExit('请运行 run_analysis.py，并用 --input 指定已预处理光谱目录。')
