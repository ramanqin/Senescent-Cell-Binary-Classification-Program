# 差异分析与小提琴图

本目录下的同名内层目录是程序所在位置。现在只做“读取已预处理光谱 → 个体内平均 → 差异检验 → 作图”；光谱清洗和批处理预处理在项目的 `data analysis` 中完成，这里不重复执行，也不再存放两套程序的副本。

## 输入与启动

`--input` 指向 `data analysis/Spectral Batch Processing` 生成的整个 `preprocessing_run_*` 目录。批处理时须保留原目录结构：

```text
preprocessing_run_时间/
├─ 年轻/个体编号/*.txt
└─ 衰老/个体编号/*.txt
```

也接受双列 CSV。文件可以无表头，也可以带本项目批处理程序默认的一行 `Raman_shift_cm-1,Intensity` 表头。每条光谱必须是 600–1800 cm⁻¹、步长 1 的 1201 个数值点，第一列波数、第二列强度；每组至少 2 个独立个体。同一个体可有多条谱，程序先取个体内平均，再让每个个体在统计中占一票。原始数据、清洗输出的 `passed` 目录和只含某一组的目录不能直接用作本程序输入。

在本目录打开终端：

```powershell
cd '.\Significance Analysis and Violin Plot Drawing'
python -m pip install -r requirements.txt
python -B -X utf8 run_analysis.py --input 'D:\预处理结果\preprocessing_run_时间' --output-root 'D:\差异结果'
```

不传 `--output-root` 时，结果放在内层目录 `result/` 下；每次运行新建时间戳子目录。输入文件只读。若出现 `运行失败.txt`，该次运行没有完成，不能用该目录得出结论。

## 文件和参数

| 文件 | 用途 |
|---|---|
| `run_analysis.py` | 检查目录、波数网格和数值，按个体平均并启动统计及导出。 |
| `differential_analysis.py` | 逐波数 Mann–Whitney U 检验、BH-FDR 校正、连续候选区间和可选峰面积检验。 |
| `compact_result.py` | 汇总文字、候选表和最终图片。 |
| `intensity_violin.py` | 入选波数点的个体强度小提琴图。 |
| `peak_violin.py` | 可选峰面积小提琴图。 |
| `intensity_heatmap.py` | 候选位点强度热图。 |
| `pvalue_heatmap.py` | 平均光谱及逐波数 p 值色带。 |
| `tests/test_linked_violin.py`、`tests/test_preprocessed_input.py` | 图表联动、预处理输入和完整导出测试。 |

`run_analysis.py` 顶部的 `RAW_P_THRESHOLD`、`VIOLINS_PER_PAGE`、`INCLUDE_AREA_VIOLIN` 是默认值；`PREPROCESSED_DATA_DIR` 默认留空。通常无需改代码，可在命令行传入 `--raw-p-threshold 0.01`、`--violins-per-page 10`、`--include-area`。找峰参数 `PEAK_*` 与 FDR 阈值 `FDR_ALPHA` 在 `differential_analysis.py` 顶部。清洗和光谱预处理参数只在 `data analysis` 对应程序里改；若改了输出波数范围或步长，本程序的输入网格要求也要同步调整。详情见[内层说明](Significance%20Analysis%20and%20Violin%20Plot%20Drawing/README.md)。

主包测试：`python -B -m unittest discover -s tests -v`。
