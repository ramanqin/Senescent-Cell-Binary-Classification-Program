# 通用PCA-SVM光谱分析

输入必须是已经预处理且波数网格完全一致的双列光谱。本程序不会自动重采样。

类别统一为：年轻=0（阴性），衰老=1（阳性）。类别目录下第一级子文件夹默认代表独立个体，同一文件夹内光谱先求平均。

批次信息会取类别目录之前的相对路径；如果类别目录直接位于输入根目录，则使用输入根目录名。因此P4和P7中同名个体不会再被合并。

每次运行的结果保存到：

```text
result_plot/analysis_run_年月日_时分秒/
├─ ROC.png
├─ predictions.csv
├─ run_manifest.csv
├─ run_parameters.json
├─ metrics.json
└─ model.joblib
```

`run_manifest.csv`逐条记录进入建模的光谱、批次、类别和独立样本编号。
