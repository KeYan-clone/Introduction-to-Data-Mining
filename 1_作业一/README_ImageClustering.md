# 无监督图像聚类项目

## 项目概述

本项目实现了一个完整的无监督图像聚类系统，用于对图像数据集进行自动分类。项目涵盖了图像特征提取、聚类算法选择和聚类效果评估三个核心部分。

## 数据集

- **位置**: `DM_2025_Dataset/Cluster/Cluster/dataset/`
- **规模**: 600张PNG图像
- **类别**: 6类 (transistor, leather, pill, bottle, tile, cable)
- **标签文件**: `cluster_labels.json`

## 技术方案

### 1.1 图像特征处理 (10%)

#### 深度学习特征提取
- **模型**: 预训练ResNet18 (ImageNet权重)
- **特征维度**: 512维
- **预处理**:
  - 调整尺寸至224×224
  - 标准化 (ImageNet均值和标准差)

#### 传统特征提取
| 特征类型   | 维度   | 描述                         |
| ---------- | ------ | ---------------------------- |
| 颜色直方图 | 96维   | RGB三通道各32个bins          |
| HOG特征    | 1764维 | 方向梯度直方图，捕获纹理信息 |

#### 特征处理流程
```
原始图像 → 特征提取 → 标准化(StandardScaler) → PCA降维 → 特征融合
```

- 深度特征: 512维 → PCA → 50维
- 传统特征: 颜色+HOG → PCA → 30维
- 融合特征: 50 + 30 = 80维

### 1.2 聚类算法选择 (10%)

本项目实现了4种聚类算法进行对比：

| 算法                    | 特点                   | 适用场景               |
| ----------------------- | ---------------------- | ---------------------- |
| **K-Means**             | 简单高效，基于距离     | 球形簇，大规模数据     |
| **Spectral Clustering** | 基于图论，捕获复杂结构 | 非凸形状的簇           |
| **Agglomerative**       | 层次聚类，可解释性强   | 发现数据层次结构       |
| **GMM**                 | 概率模型，软聚类       | 椭圆形簇，需要概率输出 |

#### 算法参数设置
```python
# K-Means
KMeans(n_clusters=6, n_init=20, max_iter=500)

# Spectral Clustering
SpectralClustering(n_clusters=6, affinity='nearest_neighbors', n_neighbors=15)

# Agglomerative
AgglomerativeClustering(n_clusters=6, linkage='ward')

# GMM
GaussianMixture(n_components=6, covariance_type='full', n_init=5)
```

### 1.3 聚类效果评估 (5%)

#### 无监督评估指标
| 指标              | 范围    | 最优值   | 描述                   |
| ----------------- | ------- | -------- | ---------------------- |
| Silhouette Score  | [-1, 1] | 越大越好 | 簇内紧密度与簇间分离度 |
| Calinski-Harabasz | [0, ∞)  | 越大越好 | 簇间方差/簇内方差      |
| Davies-Bouldin    | [0, ∞)  | 越小越好 | 簇内散度/簇间距离      |

#### 有监督评估指标 (用于验证)
| 指标     | 范围    | 描述                     |
| -------- | ------- | ------------------------ |
| NMI      | [0, 1]  | 归一化互信息             |
| ARI      | [-1, 1] | 调整兰德指数             |
| Accuracy | [0, 1]  | 匈牙利算法匹配后的准确率 |

## 文件结构

```
dataSwarm/
├── image_clustering.py      # 主程序
├── README_ImageClustering.md # 说明文档
├── clustering_results.png    # 可视化结果图
├── clustering_output.json    # 聚类输出结果
└── DM_2025_Dataset/
    └── Cluster/
        └── Cluster/
            ├── dataset/          # 图像数据集
            └── cluster_labels.json  # 真实标签
```

## 运行方法

### 环境要求
```bash
pip install torch torchvision pillow scikit-learn scikit-image matplotlib numpy
```

### 执行程序
```bash
python image_clustering.py
```

### 输出结果
1. **控制台输出**: 各算法的评估指标对比
2. **clustering_results.png**: t-SNE可视化图
3. **clustering_output.json**: 每张图片的聚类结果

## 可视化说明

程序生成的可视化图包含：
- 真实标签分布 (t-SNE降维)
- K-Means聚类结果
- Spectral聚类结果
- Agglomerative聚类结果
- GMM聚类结果
- 评估指标对比柱状图

## 技术亮点

1. **多特征融合**: 结合深度学习特征和传统特征，充分利用不同层次的图像信息
2. **多算法对比**: 实现4种主流聚类算法，便于选择最优方案
3. **全面评估**: 同时使用无监督和有监督指标，客观评价聚类质量
4. **可视化分析**: t-SNE降维直观展示聚类效果

## 依赖库版本

| 库           | 用途                 |
| ------------ | -------------------- |
| torch        | 深度学习框架         |
| torchvision  | 预训练模型和图像变换 |
| scikit-learn | 聚类算法和评估指标   |
| scikit-image | HOG特征提取          |
| matplotlib   | 可视化               |
| PIL          | 图像读取             |
| numpy        | 数值计算             |

## 作者

数据挖掘课程作业 - 图像聚类任务
