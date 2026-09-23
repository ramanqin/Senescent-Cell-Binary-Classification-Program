# 随机森林分类与邻近度降维代码

这里有分类和降维两个入口。`run_classification.py` 默认会运行三种分类模型，指定 `--models random_forest` 才只跑随机森林；`visualize_embeddings.py` 每次会同时输出 PCA 与 t-SNE，提供森林模型后再增加邻近度图。

| 文件 | 作用 |
|---|---|
| `run_classification.py` | 做个体级普通分层 5 折分类，保存逐折指标、折外预测、ROC、混淆矩阵和全数据最终模型。随机森林默认 500 棵树，使用 `balanced_subsample` 类别权重。 |
| `visualize_embeddings.py` | 绘制 PCA 和 t-SNE；传入 `--rf-model` 后，从森林叶节点计算个体间邻近度，再用 PCoA/MDS 作二维图及邻近度热图。 |
| `raman_io.py` | 读取 `组别/个体编号/*.txt`，检查统一波数轴并对同一个体的光谱求平均；两个入口共用。它不做清洗或预处理。 |

先使用已清洗、已预处理的双列 TXT，默认组目录为 `年轻`、`衰老`，每组至少 5 个独立个体。在本目录运行：

```powershell
python -m pip install -r requirements.txt
python run_classification.py --input "D:\预处理光谱" --output "D:\分类结果" --models random_forest
python visualize_embeddings.py --input "D:\预处理光谱" --output "D:\降维结果" --rf-model "D:\分类结果\random_forest\model.joblib"
```

分类折数在 `run_classification.py` 的 `CV_SPLITS`，树数和其他固定模型参数在 `build_model()`；`--seed` 可调随机种子。降维的 `--perplexity`、`--seed` 在 `visualize_embeddings.py` 命令行中设置。邻近度图使用带标签训练的全数据森林，只显示模型内部结构，不能作为独立测试的分类成绩。这个目录的两个脚本也有其他目录的副本，改共用逻辑时请同步核对。
