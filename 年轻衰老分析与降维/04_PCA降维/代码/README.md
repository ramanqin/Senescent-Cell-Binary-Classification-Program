# PCA 降维代码

`visualize_embeddings.py` 是本目录的入口。目录名虽是 PCA，脚本每次都会同时生成 PCA 和 t-SNE 的图；只有传入随机森林模型时，才额外生成邻近度图。

| 文件 | 作用 |
|---|---|
| `visualize_embeddings.py` | 对个体平均谱做二维 PCA，保存散点图、坐标和主成分解释方差；同一次运行还会做 t-SNE。可选的 `--rf-model` 用于随机森林邻近度 PCoA/MDS 及热图。 |
| `raman_io.py` | 读取 `组别/个体编号/*.txt`，核对波数轴一致性并计算个体内平均光谱。它不负责清洗、基线校正或重采样。 |

数据需先经过清洗和预处理，使用双列 TXT（可带本项目批处理的一行表头），默认组别为 `年轻`、`衰老`，每组至少 5 个独立个体。在本目录运行：

```powershell
python -m pip install -r requirements.txt
python visualize_embeddings.py --input "D:\预处理光谱" --output "D:\降维结果"
```

主要结果为 `pca_2d.png`、`pca_coordinates.csv`，同时也会得到 `tsne_2d.png`、`tsne_coordinates.csv` 和 `embedding_summary.json`。二维 PCA 在 `visualize_embeddings.py` 的 `main()` 中设置为 `n_components=2`；`--seed` 主要影响 t-SNE。PCA 是无监督展示，图上的重叠或分离不能直接当作分类准确率。若只想生成 PCA 图，目前脚本没有单独开关。

随项目附带的演示数据只有 10 个编号；用它试跑时加 `--perplexity 3`，否则默认 20 会超过样本数限制。
