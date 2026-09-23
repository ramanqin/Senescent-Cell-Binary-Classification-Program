"""联动导出差异分析、入选点小提琴图及热图。"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from peak_violin import plot_peak_violin
from intensity_violin import select_points, export_selected_violins
from intensity_heatmap import plot_intensity_heatmap
from pvalue_heatmap import plot_mean_spectrum_pvalue_heatmap


def export_compact(work, out, violins_per_page=10, include_area=False):
    summary = json.loads((work/'运行摘要.json').read_text(encoding='utf-8'))
    stats = summary['summary']
    threshold = stats['raw_p_threshold']
    point = pd.read_csv(work/'差异性结果'/'全部波数点_保留p和q.csv')
    selected = select_points(point, threshold)
    if len(selected) != stats['selected_point_count']:
        raise ValueError('本轮统计摘要和候选表数量不一致')
    selected[['Raman_shift', 'young_mean', 'old_mean', 'difference', 'p_value', 'q_value', 'direction']].rename(columns={
        'Raman_shift':'波数(cm-1)', 'young_mean':'年轻组平均强度', 'old_mean':'衰老组平均强度',
        'difference':'强度差(年轻-衰老)', 'p_value':'原始p值', 'q_value':'FDR校正q值',
        'direction':'变化方向'}).to_csv(out/'候选波数点.csv', index=False, encoding='utf-8-sig')
    regions = pd.read_csv(work/'差异性结果'/f'原始p小于{threshold:g}_连续区间.csv')
    settings = json.loads((work/'实际使用参数.json').read_text(encoding='utf-8'))
    lines = ['差异性分析结果', '',
        f"样本：年轻{stats['young_subjects']}个，衰老{stats['aging_subjects']}个。",
        f"输入已预处理光谱：{summary['processed_spectra']}条；按个体内平均后统计。",
        f"筛选：p<0.05为{stats['original_p005_count']}点；p<{threshold:g}为{stats['selected_point_count']}点，合并为{stats['interval_count']}个区间。", '',
        '候选区间（cm⁻¹）：']
    for r in regions.itertuples(index=False):
        interval = f'{r.start:g}–{r.end:g}' if r.start != r.end else f'{r.start:g}'
        lines.append(f'  {interval}：{r.direction}，最低p={r.min_p:.5g}')
    if regions.empty:
        lines.append('  无')
    lines += ['', f"FDR校正后（q<0.05）：{stats['fdr_significant_points']}个波数点。",
        f"峰面积检验：{stats['candidate_peak_count']}个峰中，p<{threshold:g}有{stats['selected_peak_count']}个。",
        '这些位置用于探索性峰归属，不等同于确定物质或衰老因果因素。', '',
        '方法：直接读取data analysis输出的预处理光谱；600–1800 cm⁻¹，步长1；个体内平均后做双侧Mann–Whitney U检验。本程序不重复清洗或预处理。',
        f"数据来源：{settings['input_directory']}",
        '文件说明：候选波数点.csv为明细；结果图.png上图为平均光谱，下图为p值筛选。']
    (out/'结果说明.txt').write_text('\n'.join(lines), encoding='utf-8-sig')
    matrix = pd.read_csv(work/'个体平均光谱矩阵.csv', index_col=0)
    x = matrix.index.to_numpy(dtype=float)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'], 'axes.unicode_minus':False})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, sharex=True)
    for group, color in [('年轻', '#2474a7'), ('衰老', '#d56c43')]:
        values = matrix.loc[:, matrix.columns.str.startswith(group+'/')].to_numpy()
        mean = values.mean(1)
        sem = values.std(1, ddof=1)/np.sqrt(values.shape[1])
        axes[0].plot(x, mean, color=color, label=f'{group}（n={values.shape[1]}）')
        axes[0].fill_between(x, mean-sem, mean+sem, color=color, alpha=.15)
    axes[0].set(title='两组平均光谱（均值 ± SEM）', ylabel='归一化强度')
    axes[0].legend()
    axes[1].plot(point.Raman_shift, -np.log10(np.clip(point.p_value, 1e-300, 1)), lw=1, color='#2474a7')
    for p, color in [(0.05, '#999999'), (threshold, '#d56c43')]:
        axes[1].axhline(-np.log10(p), ls='--', color=color, label=f'p={p:g}')
    axes[1].scatter(selected.Raman_shift, -np.log10(np.clip(selected.p_value, 1e-300, 1)), s=25, color='#d56c43', zorder=3)
    axes[1].set(title=f"原始p<{threshold:g}：{len(selected)}个候选点 / {len(regions)}个区间", ylabel=r'$-\log_{10}(p)$', xlabel=r'拉曼位移（cm$^{-1}$）')
    axes[1].legend()
    fig.savefig(out/'结果图.png', dpi=220)
    plt.close(fig)
    images, reports = export_selected_violins(matrix, selected, out, violins_per_page)
    heatmap_report = plot_intensity_heatmap(matrix, selected, out/'候选位点强度热图.png')
    mean_heatmap_report = plot_mean_spectrum_pvalue_heatmap(
        matrix, point, out/'平均光谱与p值热图.png', candidate_threshold=threshold)
    lines += ['', f'强度小提琴图：本轮筛选{len(selected)}点，绘制{len(reports)}点，不合并相邻点。',
        '图片：'+'、'.join(images),
        '每个散点是一个个体在该波数处的平均谱强度，不做面积积分，不二次清洗或预处理。',
        '灰色为健康（沿用年轻组标签），粉色为衰老；全部合格个体保留。',
        '图中p值逐点复算并与本轮差异分析核对一致；p未做多重校正，q值保留在候选表。',
        '窗口和统计筛选来自同一批数据，绘图是结果展示，不是独立验证。', '',
        f"强度热图：{heatmap_report['point_count']}个位点，{heatmap_report['subject_count']}个个体。",
        '行是候选波数点，列是按健康（年轻）、衰老分区排列的个体；颜色为每个位点内跨个体Z-score。',
        'Z-score用于显示同一位点内个体的相对高低，不代表原始强度、倍数变化或独立显著性检验。', '',
        f"平均光谱与p值热图：{mean_heatmap_report['point_count']}个波数点，{mean_heatmap_report['subject_count']}个个体。",
        '上图是全部个体的平均拉曼光谱，下图为与600–1800 cm⁻¹波数轴对齐的逐点p值色带。',
        '色带越亮表示p值越小，p≥0.05显示为黑色；标注波数为候选区间的最低p值代表点。']
    if heatmap_report['constant_points']:
        lines.append('组间无变异而显示为0的位点：' +
                     '、'.join(f'{value:g}' for value in heatmap_report['constant_points']))
    if include_area:
        valid_regions = regions[(regions.representative_shift-8 >= x.min()) &
                                (regions.representative_shift+8 <= x.max())]
        if len(valid_regions):
            area_report = plot_peak_violin(matrix, valid_regions, out/'峰面积小提琴图.png')
            lines += ['', '附加面积图：每个候选区间最小p点±8 cm⁻¹；面积p值另算。']
            for r in area_report:
                lines.append(f"  {r['window_start']:g}–{r['window_end']:g} cm⁻¹：p={r['raw_area_p']:.5g}")
        lines.append(f'超出光谱范围而跳过的面积窗口：{len(regions)-len(valid_regions)}个。')
    (out/'结果说明.txt').write_text('\n'.join(lines), encoding='utf-8-sig')
