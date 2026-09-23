# 差异分析与强度小提琴图

本程序从 `data analysis` 的批处理输出开始，不接原始光谱，也不重复做清洗、去噪、基线校正或归一化。运行顺序是：读取已预处理光谱 → 个体内平均 → 逐波数检验 → 选出候选位置 → 小提琴图和热图。清洗与预处理的方法、参数、质控报告请到项目 `data analysis` 对应目录查看。

## 输入

推荐用 `--input` 指向一个 `preprocessing_run_*` 目录，不用改程序里的路径：

```text
preprocessing_run_时间/
├─ 年轻/
│  ├─ 个体001/光谱1_预处理.txt
│  └─ 个体002/...
└─ 衰老/
   ├─ 个体101/...
   └─ 个体102/...
```

每组至少 2 个个体；每个个体目录至少 1 条 `.txt` 或 `.csv` 光谱。文件须为双列数值，第一列波数、第二列强度，600–1800 cm⁻¹ 且步长 1，共 1201 个数值点。可以无表头，也可以使用本项目批处理输出的一行 `Raman_shift_cm-1`、`Intensity` 表头。程序会拒绝不符合此网格的文件，避免误把原谱或其他步长的结果混进同一次分析。只有组目录下个体目录内的 TXT/CSV 会读入；批处理目录根部的报告文件不会被当成光谱。输入文件会核对哈希，不会改写。

## 运行与调参

在此目录执行：

```powershell
python -m pip install -r requirements.txt
python -B -X utf8 run_analysis.py --input "D:\预处理结果\preprocessing_run_时间" --output-root "D:\差异结果"
```

不传 `--output-root` 时，结果默认写入本目录 `result/`，每次建一个 `analysis_run_*` 子目录。也可在 `run_analysis.py` 顶部设置 `PREPROCESSED_DATA_DIR`，但准备上传代码时不要把个人机器路径留在文件里。

| 参数 | 位置或命令行 | 默认值 |
|---|---:|---:|
| 原始 p 值筛选阈值 | `run_analysis.py` 的 `RAW_P_THRESHOLD` / `--raw-p-threshold` | 0.01 |
| 每页小提琴图位点数 | `VIOLINS_PER_PAGE` / `--violins-per-page` | 10 |
| 是否额外出峰面积图 | `INCLUDE_AREA_VIOLIN` / `--include-area` | 否 |
| BH-FDR 阈值 | `differential_analysis.py` 的 `FDR_ALPHA` | 0.05 |
| 找峰突出度、间距、面积半宽 | `differential_analysis.py` 的 `PEAK_*` | 见代码 |

这里没有清洗和预处理参数副本。要调整那两步，请在 `data analysis` 重跑并生成新的 `preprocessing_run_*` 目录，再用这个新目录运行差异分析。若更改预处理波数范围或步长，需要相应修改 `run_analysis.py` 的 `EXPECTED_X`，并确认后续作图仍适用。

## 输出与口径

通常会得到 `结果说明.txt`、`候选波数点.csv`、`结果图.png`、`强度小提琴图*.png`、`候选位点强度热图.png`、`平均光谱与p值热图.png`。默认不输出中间谱、原始数据或历史缓存；开启 `--include-area` 后另有峰面积图。失败时写 `运行失败.txt`，该目录不能当成成功结果。

每个个体先对其多条光谱取算术平均，再作为一个统计单位。逐点使用双侧 Mann–Whitney U 检验，按原始 p 值给出探索性候选，同时在 CSV 里保留 BH-FDR 的 q 值。相邻候选点合并成区间仅为整理展示，不是区间层面的显著性检验；峰面积检验又是另一项检验。小提琴图中的“健康”是年轻组的展示名称，不代表第三组。图与筛选来自同一批数据，不是独立验证，也不能仅靠波数归属推出衰老的因果因素。

主要文件：`run_analysis.py` 为入口，`differential_analysis.py` 做统计，`compact_result.py` 汇总输出，`intensity_violin.py` 和 `peak_violin.py` 画小提琴图，`intensity_heatmap.py` 与 `pvalue_heatmap.py` 画热图。测试命令：

```powershell
python -B -m unittest discover -s tests -v
```
