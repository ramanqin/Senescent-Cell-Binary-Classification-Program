# 光谱抽样与人工质控标注

这个工具从目录中抽取 TXT 光谱，逐条显示原谱，由人标注 PASS、REVIEW 或 FAIL。它做的是光谱质量标注，不训练年轻/衰老分类模型。抽样时默认隐藏组别和个体编号；需要核对时可在界面中显示。

## 启动

在本目录运行：

```powershell
python -m pip install -r requirements.txt
python __main__.py --data "D:\你的原始光谱目录" --output "D:\你的结果目录\qc_annotations.csv"
```

Windows 也可双击 `Raman_Data_Annotation.bat`，启动后在界面里改数据目录和保存位置。若命令行不传路径，默认输入为本目录下的 `raw_data/`，输出为 `result/qc_annotations.csv`；通常应显式传入实际数据目录。只想查看扫描与抽样数量时，加 `--scan`，不会打开界面：

```powershell
python __main__.py --scan --data "D:\你的原始光谱目录" --group0-size 50 --group1-size 50
```

程序递归查找 TXT。能从路径识别的 `年轻/young` 记为 0 组，`衰老/aging` 记为 1 组；无法识别的放入 unknown 组。每组抽样数由 `--group0-size`、`--group1-size`、`--unknown-size` 控制，`-1` 表示该组全取，`0` 表示不取。第一次加载时会在标注 CSV 旁保存抽样清单，再次打开同一个 CSV 会恢复原名单。

## 文件分工

| 文件 | 用途 |
|---|---|
| `__main__.py` | 命令行入口；解析目录、抽样数量、随机种子及 `--scan`。 |
| `gui.py` | 光谱图、人工标注按钮、快捷键和标注进度。 |
| `core.py` | 扫描文件、识别组别/个体、抽样、读谱、计算质控指标、保存 CSV 和抽样清单。 |
| `state.py` | 界面状态及默认输入、输出、抽样数量。 |
| `qc_assistant.py` | 用已有人工质控标签训练小型决策树，给当前光谱提供辅助建议。 |
| `validate_dataset.py` | 单独核对文件数、个体数、点数、波数范围及读取异常。 |
| `test_core.py` | 扫描、抽样、读谱和存档测试。 |
| `test_gui_keyboard.py` | 界面快捷键行为测试。 |
| `test_qc_assistant.py` | 决策树辅助建议的测试。 |

`Raman_Data_Annotation.bat` 是 Windows 启动脚本，`requirements.txt` 列出依赖。

## 参数在哪改

抽样配额、`balanced/random` 方式和随机种子可以在界面里改，也可在启动命令中传入 `--group0-size`、`--group1-size`、`--unknown-size`、`--strategy` 和 `--seed`。命令行默认值在 `__main__.py`；界面直接调用时使用 `state.py` 的 `ProcessState` 默认值。

人工质控指标的算法在 `core.py` 的 `calculate_qc_metrics()`；异常原因和快捷键映射在 `gui.py`。辅助决策树的最低标注量、每类最低数量和高置信阈值在 `qc_assistant.py` 的 `MIN_TRAINING_SAMPLES`、`MIN_SAMPLES_PER_CLASS`、`HIGH_CONFIDENCE`。这些是质控建议参数，不改变年轻/衰老标签。

单独做数据检查可运行：

```powershell
python validate_dataset.py "D:\你的原始光谱目录"
python -B -m unittest discover -v
```

标注 CSV 每行对应一条光谱；正式分类时应按个体划分训练和验证，避免同一个体的多条光谱同时进入两边。
