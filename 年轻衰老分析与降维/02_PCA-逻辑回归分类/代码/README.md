# PCA-逻辑回归分类代码

这个文件夹的分类入口与 PCA-SVM、随机森林目录中的同名脚本一致。默认运行三种模型；只运行本目录对应的方法时使用 `--models logistic`。

| 文件 | 作用 |
|---|---|
| `run_classification.py` | 分类入口。按个体做普通分层 5 折交叉验证，输出每折结果、折外预测、指标、ROC、混淆矩阵及全数据最终模型。逻辑回归路线在每个训练折内拟合 PCA、标准化和逻辑回归。 |
| `raman_io.py` | 读取 `组别/个体编号/*.txt`，检查所有光谱的波数网格，将同一个体的重复光谱求平均，返回个体级数据与标签。它不会重做光谱清洗或插值。 |

输入为已清洗、已预处理的双列 TXT，默认组名 `年轻`、`衰老`；每组至少 5 个独立个体。在本目录运行：

```powershell
python -m pip install -r requirements.txt
python run_classification.py --input "D:\预处理光谱" --output "D:\分类结果" --models logistic
```

`run_classification.py` 的 `CV_SPLITS=5` 是折数；`build_model()` 中逻辑回归默认为 `C=0.001`、L2 正则、`saga` 求解器和类别权重平衡。`--pca-variance` 调整训练折内 PCA 的保留方差，`--seed` 调整随机种子。路径、类别名等可从 `--config` 的 JSON 读取；模型固定参数仍要在 `build_model()` 中修改。

结果在输出目录的 `pca_logistic_regression/` 下。五折指标来自折外预测；`model.joblib` 用全部输入个体另行拟合，不能当成验证集模型。修改共用代码时，应核对其他分类目录的副本。
