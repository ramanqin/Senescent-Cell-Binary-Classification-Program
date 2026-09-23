# PCA-SVM 分类代码

这个文件夹可以单独运行。虽然目录名是 PCA-SVM，`run_classification.py` 默认会运行 SVM、逻辑回归和随机森林三种模型；只运行 SVM 时加 `--models svm`。

| 文件 | 作用 |
|---|---|
| `run_classification.py` | 分类入口。构建模型，以个体为单位做普通分层 5 折交叉验证，保存逐折结果、折外预测、指标、ROC、混淆矩阵和全数据最终模型。SVM 路线在每个训练折内拟合 PCA、标准化及线性核 SVM。 |
| `raman_io.py` | 读取 `组别/个体编号/*.txt`，核对波数轴是否一致，将同一个体的多条光谱求平均，并生成模型使用的矩阵、标签和个体编号。它不负责清洗、预处理或重采样。 |

输入应为已清洗、已预处理的双列 TXT，两组目录默认为 `年轻` 和 `衰老`，每组至少 5 个独立个体。在本目录运行：

```powershell
python -m pip install -r requirements.txt
python run_classification.py --input "D:\预处理光谱" --output "D:\分类结果" --models svm
```

参数位置：`run_classification.py` 顶部的 `CV_SPLITS=5` 控制折数；`build_model()` 中 SVM 默认为线性核、`C=0.01`、`class_weight="balanced"`。PCA 保留方差可用 `--pca-variance` 调整，随机种子用 `--seed` 调整。`--config` 可以提供路径等设置，但不会覆盖 `build_model()` 中写定的模型参数。

结果在输出目录的 `pca_svm/` 下；`model.joblib` 是用全部个体重新拟合的最终模型，不参与五折指标计算。另两份分类目录有同名脚本副本，调整共用模型参数时要同步检查。
