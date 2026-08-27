from sklearn.tree import DecisionTreeClassifier, export_text              #创建决策树分类器，并将训练后的树导出为文本规则
from sklearn.model_selection import StratifiedKFold, cross_val_predict    #分层交叉验证工具；当前版本尚未实际使用
from sklearn.metrics import confusion_matrix                              #计算二分类混淆矩阵

import pandas as pd                                                        #读取、筛选和合并CSV表格
import numpy as np                                                         #NumPy当前未直接使用，可作为后续数值处理工具
from pathlib import Path                                                   #搜索目录中的CSV文件并处理路径

mode = '1447'                                                              

# region io：定位本次实验所需的输入文件

# 人工标注结果目录；CSV中应包含文件绝对路径和qualified分类标签。
marking_folder_path = r'D:\coding_environment\programs\020_w_working_place_for_Ecoli_programs\working_data\sampling_original_data_for_filter_paras\attribute_record\result_marking'
# 光谱特征目录；CSV通常由parameter_finding.py批量提取得到。
attribute_folder_path = r'D:\coding_environment\programs\020_w_working_place_for_Ecoli_programs\working_data\sampling_original_data_for_filter_paras\attribute_record\attribute'

# 只搜索当前目录下的CSV，不递归进入子目录。
all_marking_file_path = list ( Path( marking_folder_path).glob('*.csv'))    #所有候选标注表
all_attribute_file_path = list ( Path( attribute_folder_path).glob('*.csv'))  #所有候选特征表
present_marking_path = ''                                                  #最终选中的标注表路径
present_attribute_path = ''                                                #最终选中的特征表路径

for marking in all_marking_file_path:                                      #逐个检查候选标注表

    # 将不带扩展名的文件名按下划线分段，要求其中一个分段与mode完全相同。
    if mode in str(marking.stem).split('_'):

        present_marking_path = str ( marking )                             #保存匹配路径

if not present_marking_path:                                               #没有匹配标注表时无法继续训练

    raise ValueError ('no marking file fits current mode!')
    
for attribute in all_attribute_file_path:                                  #用相同规则寻找本模式的特征表

    if mode in str(attribute.stem).split('_'):

        present_attribute_path = str ( attribute )                         #保存匹配的特征表路径

if not present_attribute_path:                                             #没有匹配特征表时主动终止

    raise ValueError ('no attribute file fits current mode!')



# endregion

# 读取人工标注表：第一行作为列名，内容按UTF-8编码、逗号分隔。
present_marking_df = pd.read_csv    (
                                    present_marking_path ,
                                    sep=",",
                                    encoding="utf-8",
                                    header=0
                                    )

# 读取parameter_finding.py输出的光谱特征表。
present_attribute_df = pd.read_csv  (
                                    present_attribute_path ,
                                    sep=",",
                                    encoding="utf-8",
                                    header=0
                                    )

# 根据原始光谱文件路径对两张表进行内连接，只保留两边都能匹配到的样本。
merge_df = pd.merge (
                    present_marking_df , 
                    present_attribute_df ,
                    left_on = 'file_absolute_path' ,                        #标注表中的路径字段
                    right_on = 'path' ,                                    #特征表中的路径字段
                    how = 'inner'                                          #内连接会丢弃任一侧缺失的样本
                    )

# 打印连接前后的样本数量，用于检查路径能否正确匹配。
print("特征表数量:", len(present_attribute_df))
print("标注表数量:", len(present_marking_df))
print("合并后数量:", len(merge_df))

# 指定决策树的输入特征；顺序也会用于后面的可读规则输出。
features = [
            "Finger_SNR",                                                   #指纹区信噪比
            "Finger_3_Height",                                              #指纹区最大突出峰的相对峰高
            "Finger_3_Width",                                               #指纹区最大突出峰的宽度

            "Silence_SNR",                                                  #静默区信噪比
            "Silence_STD",                                                  #静默区强度标准差

            "CH_SNR",                                                       #C-H区信噪比
            "CH_Peak_Height",                                               #C-H主峰相对峰高
            "CH_Peak_Width",                                                #C-H主峰宽度

            "Peak_Height_Ratio_3",                                          #C-H主峰与最大指纹峰的绝对突出度比值
            ]

X = merge_df[features].copy()                                               #构建模型输入特征矩阵X，并复制以避免修改原表
y = merge_df["qualified"].astype(int)                                      #把人工标注转为整数类别标签y（通常0=不合格，1=合格）

# 可选调试代码：查看每个特征的缺失值数量及两类样本分布。
# print ('空值检测')
# print(X.isna().sum())
# print(y.value_counts())

# 创建决策树分类器，通过限制树深和叶节点样本数降低过拟合程度。
tree = DecisionTreeClassifier   (
                                max_depth=4,                                #树分裂到4层
                                min_samples_leaf=5,                         #每个叶节点包含5个训练样本
                                class_weight={0: 2, 1: 1},                  #类别0错误的惩罚权重是类别1的两倍
                                random_state=42                             #固定随机种子
                                )

tree.fit(X, y)                                                             #使用全部合并样本训练决策树

# 将每个分裂节点的“特征 <= 阈值”结构转换为可阅读的文本规则。
rules = export_text (
                    tree,
                    feature_names=features,                                #用真实特征名替代X[0]等编号
                    decimals=4                                             #规则阈值保留4位小数
                    )

print(rules)                                                               #在控制台打印决策树筛选规则

# 直接在训练集X上预测；该结果用于描述训练拟合情况，不等同于独立测试集性能。
y_pred = tree.predict(X)

# 生成2×2混淆矩阵，并依次拆分为真负、假正、假负和真正。
tn, fp, fn, tp = confusion_matrix(
    y,                                                                      #真实人工标注
    y_pred,                                                                 #决策树预测标签
    labels=[0, 1]                                                           #固定类别顺序，确保ravel后的变量顺序稳定
).ravel()

fpr = fp / (fp + tn)                                                       #假阳性率：实际不合格却预测为合格的比例
fnr = fn / (fn + tp)                                                       #假阴性率：实际合格却预测为不合格的比例

# 输出混淆矩阵的四项计数，以及百分比格式的两种错误率。
print("TP:", tp)
print("FP:", fp)
print("TN:", tn)
print("FN:", fn)
print("FPR:", f"{fpr:.2%}")
print("FNR:", f"{fnr:.2%}")
