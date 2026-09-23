# 年轻/衰老分类与降维

这里有三个分类目录、两个降维目录。目录名方便查找结果；代码本身是可复用的：三个分类目录中的 `run_classification.py` 内容相同，默认会把三种模型都跑一遍，想只跑一种就加 `--models`。三个目录中的 `visualize_embeddings.py` 也相同，每次都会画 PCA 和 t-SNE；传入随机森林模型后还会画邻近度图。

## 输入数据

使用经过清洗和预处理、波数轴完全一致的双列 TXT：

```text
数据目录/
├─ 年轻/
│  ├─ 个体01/*.txt
│  └─ 个体02/*.txt
└─ 衰老/
   ├─ 个体01/*.txt
   └─ 个体02/*.txt
```

每个体目录内的多条光谱先取平均，再以个体为单位建模。每类至少需要 5 个独立个体。当前分类代码要求各光谱的波数点逐点一致，不会自动插值。年轻为 0 类，衰老为 1 类。

## 怎么运行

以下命令在本目录运行，路径换成自己的数据和结果目录：

```powershell
python -m pip install -r ".\01_PCA-SVM分类\代码\requirements.txt"

python ".\01_PCA-SVM分类\代码\run_classification.py" `
  --input "D:\预处理光谱" --output ".\新结果\分类" `
  --models svm logistic random_forest

python ".\04_PCA降维\代码\visualize_embeddings.py" `
  --input "D:\预处理光谱" --output ".\新结果\降维" `
  --rf-model ".\新结果\分类\random_forest\model.joblib"
```

只跑 SVM 可用 `--models svm`；逻辑回归和随机森林分别用 `logistic`、`random_forest`。降维只需 PCA 和 t-SNE 时，省略 `--rf-model`。t-SNE 的 `--perplexity` 默认 20，必须小于参与分析的个体数；样本少时可调低，例如 5 或 10。

分类输出中，`model_summary.csv` 汇总三种模型；各模型子目录有 `fold_results.json`、`predictions.csv`、`metrics.json`、ROC、混淆矩阵和最终全数据模型。五折分数来自折外预测，最终模型另外用全部输入个体拟合。降维输出在指定目录中，包含 `pca_2d.png`、`tsne_2d.png`、坐标 CSV、`embedding_summary.json`；传入森林模型时增加 `rf_proximity_pcoa.png` 和邻近度热图。

## Python 文件与参数位置

| 位置 | 用途 |
|---|---|
| `01_PCA-SVM分类/代码/run_classification.py` | 三种模型的固定参数普通分层 5 折、指标、ROC、混淆矩阵及模型保存。`02`、`03` 目录有同名副本。 |
| `01_PCA-SVM分类/代码/raman_io.py` | 读入双列光谱、核对波数轴、按个体平均。其余四个目录各有一份副本。 |
| `03_随机森林分类与邻近度降维/代码/visualize_embeddings.py` | PCA、t-SNE，以及可选的随机森林邻近度 PCoA 和热图。`04`、`05` 目录有同名副本。 |

分类默认参数写在各份 `run_classification.py` 的 `build_model()` 和 `CV_SPLITS`：SVM 用线性核、C=0.01；逻辑回归用 C=0.001；随机森林 500 棵树。PCA 保留方差可用 `--pca-variance` 改，随机种子用 `--seed` 改。脚本中的 `--config` 也可读取 JSON 的输入、输出、类别、种子及 PCA 方差；它不会覆盖 `build_model()` 中写死的模型参数。若要调整 C、树数或折数，需修改脚本，并同步修改另外两份分类副本，以免三个目录跑出不同设置。

降维参数在 `visualize_embeddings.py` 的命令行和 `main()` 中：`--perplexity`、`--seed` 可直接传；PCA 保留95%方差作为 t-SNE 前处理写在 `pca95` 处。若改 t-SNE 的其它参数，也应同步三份副本。随机森林邻近度基于已训练的监督模型，图上的分离程度不等于独立验证成绩。

`01_PCA-SVM分类/说明.md` 记录的是历史分析背景，其中提到的旧结果不能直接当作当前普通 5 折代码的输出；新分析请看本次新生成的 `metrics.json`。这里的验证是个体级分层 5 折，并未按实验批次留出验证。
