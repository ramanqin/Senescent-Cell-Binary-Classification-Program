# PCA-SVM 图形分析

这一子目录的程序实际放在 `PCA_SVM_GUI/`。它读取已经预处理、波数网格一致的光谱，按个体求平均后做年轻/衰老二分类。验证方式是固定参数的普通分层 5 折；PCA 和标准化都在各折的训练数据中拟合。

## 启动

先进入内层程序目录：

```powershell
cd .\PCA_SVM_GUI
python -m pip install -r requirements.txt
python gui.py
```

Windows 下也可双击内层目录的 `launch_gui.bat`。在界面里选输入、输出目录，核对年轻和衰老的目录别名后运行。要求每类至少有 5 个独立个体；不同光谱的波数轴不一致时程序会报错，需要先用批处理程序统一网格。

命令行方式读取 `PCA_SVM_GUI/config.json`：

```powershell
python pca_svm_analysis.py config.json
```

先把配置中的 `input_dir`、`output_dir` 占位路径改成实际地址。输出会在目标目录中新建 `analysis_run_时间/`，包括 ROC、混淆矩阵、每折结果、逐个体预测、运行参数和全数据拟合的 `model.joblib`。报告中的交叉验证指标来自折外预测，最终模型单独用于后续使用。

## Python 文件

| 文件 | 用途 |
|---|---|
| `PCA_SVM_GUI/gui.py` | 图形界面；选择目录和设置 PCA、SVM 参数。 |
| `PCA_SVM_GUI/pca_svm_analysis.py` | 读谱、按个体汇总、普通分层交叉验证、计算指标、出图和保存模型；也是命令行入口。 |
| `PCA_SVM_GUI/test_single_cv.py` | 固定参数 5 折、每折拟合和参数校验测试。 |
| `PCA_SVM_GUI/test_grid_validation.py` | 输入波数网格、个体识别及输出清单测试。 |

`launch_gui.bat` 启动界面，`requirements.txt` 是内层目录的依赖文件。

## 参数在哪改

图形界面里可改 PCA 保留方差、SVM 核函数、C 和 gamma；折数在界面固定为 5。命令行参数集中在 `PCA_SVM_GUI/config.json`，其中默认是 `pca_variance=0.95`、`svm_kernel=linear`、`svm_c=0.01`、`class_weight=balanced`、`cv_splits=5`。输入路径、类别目录别名和随机种子也在这个文件中。代码默认值与校验逻辑在 `pca_svm_analysis.py` 的 `normalize_config()`；修改配置后，本轮使用的值会写入 `run_parameters.json`。

这里不做内层搜索。默认 C 值参考过往 P2/P9 分析；如果在同一批数据上反复比较参数，再报告这批数据的 5 折分数，应说明这一选择过程。

检查命令（在内层目录执行）：

```powershell
python -B -m unittest discover -v
```
