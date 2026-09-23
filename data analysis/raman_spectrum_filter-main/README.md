# 光谱清洗

这个程序按指纹区（600–1800 cm⁻¹）、静默区（1800–2700 cm⁻¹）和 C–H 区（2700–3200 cm⁻¹）的质量指标筛选原始 TXT 光谱。输入和输出目录在界面里选；建议先用几条光谱核对阈值，再处理整批数据。

## 怎么启动

在本目录打开终端：

```powershell
python -m pip install -r requirements.txt
python __main__.py
```

`__main__.py` 会打开清洗界面。界面里选择原始数据目录和另一个输出目录，确认参数后开始运行。`spectrum_batch_viewer.py` 是抽样看谱工具，可单独用 `python spectrum_batch_viewer.py` 启动；Windows 下也可双击 `启动光谱抽样查看器.bat`。它只负责查看，不执行清洗。

每次清洗会新建 `cleaning_run_时间/`：`passed/` 放通过的光谱，`failed/` 放未通过或处理出错的光谱；`run_manifest.csv` 逐条记录指标和失败规则，`run_parameters.json` 保存本次参数。下一步预处理应选择这次运行的 `passed/`。

## 文件分工

| 文件 | 用途 |
|---|---|
| `__main__.py` | 桌面界面入口。 |
| `gui.py` | 选择目录、设置/保存/载入阈值、显示运行进度。 |
| `state.py` | `Parameter_State` 默认参数，包括各波段边界、SNR 和峰形阈值。 |
| `worker.py` | 在后台线程调用清洗，避免界面卡住。 |
| `core.py` | 遍历光谱、判断是否通过、复制到 `passed/failed/` 并生成清单。 |
| `parameter_finding.py` | 计算 SNR、噪声和峰形指标；直接运行时可批量导出特征表，是独立的参数探索工具。 |
| `io.py` | 独立的目录选择辅助函数，清洗主入口不直接调用。 |
| `spectrum_batch_viewer.py` | 按文件夹抽样，逐条查看 TXT 光谱。 |
| `decisiontree.py` | 旧版人工标注/特征表的决策树实验，使用文件内写死的外部路径；不在清洗主流程中调用。 |
| `__init__.py` | 包版本及导入入口。 |
| `test_cleaning_logic.py` | 清洗规则和输出清单的测试。 |

`parameter_record/` 下的 JSON 是过往试验保存的参数记录，不会在启动时自动套用。要用其中某份参数，在界面里手动载入并核对来源。

## 参数在哪改

日常调参直接用 `gui.py` 的界面，保存后得到一份 JSON。程序自带默认值在 `state.py` 的 `Parameter_State`；判定规则在 `core.py` 的 `_evaluate_result()`。例如 `Finger_Min_SNR`、`Silence_Min_SNR`、`CH_Min_SNR` 默认都是 35。调整阈值后应查看 `run_manifest.csv` 中实际被剔除的文件和 `failed_rules`，不能只看总体通过率。

`parameter_finding.py` 单独运行时，参数还会在该文件顶部的 `if __name__ == "__main__"` 块里赋值，与清洗界面中的设置不是同一个入口。`decisiontree.py` 的外部 CSV 路径与树参数也只影响旧实验脚本。

## 检查

```powershell
python -B -m unittest discover -v
```

项目自带的 `.venv` 是原电脑的环境。换电脑时重新建环境并安装 `requirements.txt` 即可。
