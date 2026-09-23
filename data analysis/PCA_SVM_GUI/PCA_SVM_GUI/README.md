# PCA-SVM光谱分析：普通5折交叉验证

2026-09-16起，本程序改为固定参数的单层分层5折验证，不再执行内层搜索或GridSearchCV。图形界面固定显示5折。

每轮4份训练、1份留出评价，轮流覆盖全部编号。PCA和StandardScaler在每轮训练数据内拟合；最终另用全部输入编号拟合一个固定参数模型用于后续预测。

默认SVM为linear核、C=0.01、class_weight=balanced；PCA保留95%方差。该默认配置参考此前P2/P9分析，不属于从未参考本数据的独立预注册方案。若参数曾在同一数据上选取，普通5折不能消除历史选择影响。不要反复查看5折成绩后挑最好参数并称为独立验证。

此程序仍是编号级分层验证，不是按批次分组验证，不能排除类别与批次的混杂。每类编号少于折数时直接报错，不自动减少折数。

命令行使用`cv_splits`、`svm_kernel`、`svm_c`、`svm_gamma`配置固定参数。旧配置的`outer_splits`兼容映射为`cv_splits`；`inner_splits`、`c_values`、`gamma_values`和`n_jobs`不再参与分析，运行配置会在`ignored_legacy_fields`记录它们。旧网格值不会自动转成固定参数。

当前目录仅保留固定参数的普通5折版本，原嵌套验证备份已删除。

## 启动前端

双击 `launch_gui.bat`。该脚本会进入程序所在目录并运行 `gui.py`；如果启动失败，命令窗口会保留具体错误信息。

输入必须是已经预处理且波数网格完全一致的双列光谱。本程序不会自动重采样。

类别统一为：年轻=0（阴性），衰老=1（阳性）。类别目录下第一级子文件夹默认代表独立个体，同一文件夹内光谱先求平均。

批次信息会取类别目录之前的相对路径；如果类别目录直接位于输入根目录，则使用输入根目录名。因此P4和P7中同名个体不会再被合并。

每次运行的结果保存到：

```text
result_plot/analysis_run_年月日_时分秒/
├─ ROC.png
├─ 混淆矩阵.png
├─ predictions.csv
├─ fold_results.json
├─ 每折结果.csv
├─ run_manifest.csv
├─ run_parameters.json
├─ metrics.json
└─ model.joblib
```

`run_manifest.csv`逐条记录进入建模的光谱、批次、类别和独立样本编号。

`predictions.csv`包含每个编号所属的`cv_fold`和折外预测。`model.joblib`保持原程序的字典封装，使用`joblib.load(path)["model"]`取得完整PCA/标准化/SVM流水线；参数字段改为`fixed_parameters`，不再标为“最佳参数”。文件中的最终全数据模型不是计算报告成绩时使用的五个折内模型。
