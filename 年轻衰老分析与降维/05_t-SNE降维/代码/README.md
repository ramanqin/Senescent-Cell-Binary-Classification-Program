# t-SNE 降维代码

`visualize_embeddings.py` 与 PCA、随机森林邻近度目录中的同名脚本一致。它每次先输出二维 PCA，再做 t-SNE；目录名不表示只运行 t-SNE。

| 文件 | 作用 |
|---|---|
| `visualize_embeddings.py` | 将个体平均谱先用 PCA 保留 95% 方差，再标准化并做二维 t-SNE；保存散点图、坐标及参数摘要。它也输出二维 PCA 图，提供 `--rf-model` 时还输出森林邻近度图。 |
| `raman_io.py` | 读取 `组别/个体编号/*.txt`，检查统一波数轴并对同一个体的多条光谱求平均，生成降维使用的个体级矩阵。它不会进行光谱预处理。 |

输入为已清洗、已预处理的双列 TXT，默认组目录为 `年轻`、`衰老`，每组至少 5 个独立个体。在本目录运行：

```powershell
python -m pip install -r requirements.txt
python visualize_embeddings.py --input "D:\预处理光谱" --output "D:\降维结果" --perplexity 20 --seed 42
```

`--perplexity` 必须大于 0 且小于个体总数；样本少时可改为 5 或 10。PCA 保留 95% 方差的前处理在 `visualize_embeddings.py` 的 `pca95` 处；t-SNE 的学习率、迭代数等在 `main()` 中设置。主要看 `tsne_2d.png`、`tsne_coordinates.csv` 和 `embedding_summary.json`。t-SNE 是探索性可视化，整体嵌入后的图不能代替交叉验证或外部测试。若只想生成 t-SNE 图，目前脚本没有单独开关。

随项目附带的演示数据共 10 个编号，用它试跑请设 `--perplexity 3`；上面的 20 只适用于更多编号的数据。
