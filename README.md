# 拉曼光谱衰老分析项目

第一次接手建议先看[快速开始](快速开始.md)：有随项目附带的合成演示数据、可直接复制的命令，以及每一步该检查的输出。演示数据只用来确认程序能衔接，不能用于研究结论。

项目分为四部分：`data analysis` 负责光谱清洗、批处理预处理及辅助标注；`年轻衰老分析与降维` 用当前输入数据重新建模；`Significance Analysis and Violin Plot Drawing` 负责逐波数差异检验及作图；`P2P9固定预测模型` 用以前训练好的模型直接预测新批次。后三部分都使用 `data analysis` 的预处理输出，不再各自清洗光谱。**新批次的独立外部测试应使用固定模型预测入口，不要误用重新训练的分类脚本。** `data analysis/PCA_SVM_GUI` 是可选的图形版 PCA-SVM 入口，不是主流程中必须额外完成的一步。

## 建议的分析顺序

1. 整理原始光谱，确认组别和个体编号无误。需要人工核对标签时，先使用 `Raman_Data_Annotation`。
2. 用 `data analysis/raman_spectrum_filter-main` 清洗原谱，检查本轮报告，将 `cleaning_run_*/passed` 作为下一步输入。
3. 用 `data analysis/Spectral Batch Processing` 批量预处理，保留组别/个体目录层级；默认截取 600–1800 cm⁻¹、步长 1 cm⁻¹。检查输出目录中的处理报告及失败文件。
4. 若要在这批有标签数据上重新建模，把 `preprocessing_run_*` 交给分类/降维；若要研究候选位点，交给差异分析。这两类结果回答不同问题，不要混作一种证据。
5. 若要用已有 P2/P9 模型预测新的批次，把该批次按相同设置预处理后交给 `P2P9固定预测模型`；这一步不重新训练。其训练数据中采集设置与年轻/衰老标签重合，解释外测结果时必须考虑这一限制。

## 数据目录要求

原始数据建议按下面的结构放置。`个体编号` 是统计和划分数据时使用的独立单位；同一个体的多条光谱放在同一目录，不要把单条光谱当成不同个体。

```text
原始数据/
├─ 年轻/
│  ├─ 个体001/
│  │  ├─ 光谱1.txt
│  │  └─ 光谱2.txt
│  └─ 个体002/...
└─ 衰老/
   ├─ 个体101/...
   └─ 个体102/...
```

原始 TXT 应为无表头的两列数字（波数、强度），具体覆盖范围和格式以清洗、预处理程序各自的 README 为准。清洗通过的文件位于 `cleaning_run_*/passed/年轻/个体编号/` 和 `.../衰老/个体编号/`。预处理时开启“保留原目录结构”，得到：

```text
preprocessing_run_时间/
├─ 年轻/个体编号/*_预处理.txt
├─ 衰老/个体编号/*_预处理.txt
├─ run_manifest.csv
└─ run_parameters.json
```

分类与差异分析都以整个 `preprocessing_run_时间` 为输入目录，不是以 `passed`、单个组别目录或原始光谱目录为输入。差异分析要求两组各至少 2 个独立个体，每条谱为两列数值、600–1800 cm⁻¹、步长 1 的 1201 点；可以有批处理程序默认的一行表头。分类的样本量和划分要求见分类 README。若修改预处理波数范围或步长，需要同步检查下游程序的输入要求。

## 各部分说明

- [光谱清洗](data%20analysis/raman_spectrum_filter-main/README.md)
- [光谱批处理](data%20analysis/Spectral%20Batch%20Processing/README.md)
- [PCA-SVM 图形分析](data%20analysis/PCA_SVM_GUI/README.md)
- [人工质控标注](data%20analysis/Raman_Data_Annotation/README.md)
- [年轻/衰老分类与降维](年轻衰老分析与降维/README.md)
- [差异分析与小提琴图](Significance%20Analysis%20and%20Violin%20Plot%20Drawing/README.md)
- [P2/P9 固定预测模型](P2P9固定预测模型/README.md)
- [合成演示数据](示例数据/README.md)

项目由其他目录整理而来，旧版本中的 `D:\raw_data`、`D:\data analysis` 等绝对路径不能直接照搬。用界面选择目录，或在命令行传入实际路径。例如，在项目根目录运行差异分析：

```powershell
python ".\Significance Analysis and Violin Plot Drawing\Significance Analysis and Violin Plot Drawing\run_analysis.py" --input "D:\分析结果\preprocessing_run_时间" --output-root "D:\差异结果"
```

建议使用 Python 3.11 或更高版本。第一次运行只需按[快速开始](快速开始.md)在项目根目录建立自己的虚拟环境并安装根目录 `requirements.txt`；子目录依赖文件供单独使用某个模块时安装。清洗目录里现有的 `.venv` 属于原电脑，不应复制使用；`.gitignore` 会在上传代码时忽略它，但不会自动删除本机文件。各程序的启动方式和参数位置见对应 README。输出目录应与输入数据目录分开；程序不会改写原始光谱。
