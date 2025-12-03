"""
无监督图像聚类任务
====================
1.1 图像特征处理方法
1.2 聚类算法选择
1.3 聚类效果评估
"""

import os
import json
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1.1 图像特征处理 (10% Score)
# ============================================================================
"""
图像特征提取方法选择说明：
1. 传统方法：
   - 颜色直方图：提取图像的颜色分布特征
   - HOG (Histogram of Oriented Gradients)：提取边缘和纹理特征
   - SIFT/SURF：局部特征描述符

2. 深度学习方法：
   - 使用预训练CNN模型(如ResNet, VGG)提取高级语义特征
   - 优点：能够捕获更丰富的图像语义信息

本任务采用：
- 主方法：预训练ResNet18提取深度特征（2048维）
- 辅助方法：颜色直方图 + HOG特征作为对比
- 使用PCA进行降维，减少计算复杂度并去除噪声
"""

import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 脚本目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据路径
DATA_DIR = os.path.join(SCRIPT_DIR, "Cluster", "Cluster", "dataset")
LABEL_FILE = os.path.join(SCRIPT_DIR, "Cluster", "Cluster", "cluster_labels.json")

# 加载真实标签（用于评估）
with open(LABEL_FILE, 'r') as f:
    true_labels_dict = json.load(f)

# 获取所有图片文件名并排序
image_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.png')])
print(f"图片数量: {len(image_files)}")

# 获取真实标签
true_labels = [true_labels_dict[f] for f in image_files]
unique_labels = list(set(true_labels))
print(f"类别数量: {len(unique_labels)}")
print(f"类别: {unique_labels}")

# 将标签转换为数字
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
true_labels_numeric = np.array([label_to_idx[l] for l in true_labels])

# ============================================================================
# 方法1：使用预训练ResNet18提取深度特征
# ============================================================================
print("\n" + "="*60)
print("1.1 图像特征处理")
print("="*60)

def extract_deep_features(image_files, data_dir):
    """使用预训练ResNet18提取图像的深度特征"""
    print("\n使用预训练ResNet18提取深度特征...")

    # 加载预训练模型
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # 移除最后的全连接层，获取特征
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    features = []
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(data_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model(img_tensor)
        features.append(feature.cpu().numpy().flatten())

        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{len(image_files)} 张图片")

    features = np.array(features)
    print(f"  深度特征维度: {features.shape}")
    return features

def extract_color_histogram(image_files, data_dir, bins=32):
    """提取颜色直方图特征"""
    print("\n提取颜色直方图特征...")
    features = []

    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)

        # 计算RGB三个通道的直方图
        hist_features = []
        for channel in range(3):
            hist, _ = np.histogram(img_array[:, :, channel], bins=bins, range=(0, 256))
            hist = hist / hist.sum()  # 归一化
            hist_features.extend(hist)

        features.append(hist_features)

    features = np.array(features)
    print(f"  颜色直方图特征维度: {features.shape}")
    return features

def extract_hog_features(image_files, data_dir):
    """提取HOG特征"""
    from skimage.feature import hog
    from skimage.color import rgb2gray
    from skimage.transform import resize

    print("\n提取HOG特征...")
    features = []

    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)

        # 转换为灰度图并调整大小
        gray = rgb2gray(img_array)
        gray_resized = resize(gray, (128, 128))

        # 提取HOG特征
        hog_feature = hog(gray_resized, orientations=9, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), visualize=False)
        features.append(hog_feature)

    features = np.array(features)
    print(f"  HOG特征维度: {features.shape}")
    return features

# 提取各种特征
deep_features = extract_deep_features(image_files, DATA_DIR)
color_features = extract_color_histogram(image_files, DATA_DIR)
hog_features = extract_hog_features(image_files, DATA_DIR)

# 特征标准化
scaler_deep = StandardScaler()
deep_features_scaled = scaler_deep.fit_transform(deep_features)

scaler_color = StandardScaler()
color_features_scaled = scaler_color.fit_transform(color_features)

scaler_hog = StandardScaler()
hog_features_scaled = scaler_hog.fit_transform(hog_features)

# PCA降维
print("\n使用PCA进行特征降维...")
pca_deep = PCA(n_components=50, random_state=42)
deep_features_pca = pca_deep.fit_transform(deep_features_scaled)
print(f"  深度特征PCA后维度: {deep_features_pca.shape}")
print(f"  深度特征PCA解释方差比例: {sum(pca_deep.explained_variance_ratio_):.4f}")

pca_combined = PCA(n_components=30, random_state=42)
combined_features = np.hstack([color_features_scaled, hog_features_scaled])
combined_features_pca = pca_combined.fit_transform(combined_features)
print(f"  传统特征(颜色+HOG) PCA后维度: {combined_features_pca.shape}")

# 融合特征
fusion_features = np.hstack([deep_features_pca, combined_features_pca])
print(f"  融合特征维度: {fusion_features.shape}")

# ============================================================================
# 1.2 聚类算法选择 (10% Score)
# ============================================================================
"""
聚类算法选择说明：
1. K-Means: 经典划分聚类，适合球形簇
2. Spectral Clustering: 谱聚类，适合非凸形状的簇
3. Agglomerative Clustering: 层次聚类，可以发现数据的层次结构
4. DBSCAN: 基于密度的聚类，不需要预先指定簇数量
5. Gaussian Mixture Model (GMM): 基于概率模型，可以处理椭圆形簇

选择理由：
- 由于已知有6个类别，选择需要指定簇数量的算法更合适
- 图像特征通常具有复杂的分布，谱聚类表现较好
- 使用多种算法进行对比，选择最优结果
"""

print("\n" + "="*60)
print("1.2 聚类算法选择与实现")
print("="*60)

from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture

n_clusters = len(unique_labels)  # 6个类别

def apply_kmeans(features, n_clusters):
    """K-Means聚类"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    labels = kmeans.fit_predict(features)
    return labels, kmeans

def apply_spectral(features, n_clusters):
    """谱聚类"""
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=42,
                                  affinity='nearest_neighbors', n_neighbors=15)
    labels = spectral.fit_predict(features)
    return labels, spectral

def apply_agglomerative(features, n_clusters):
    """层次聚类"""
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agg.fit_predict(features)
    return labels, agg

def apply_gmm(features, n_clusters):
    """高斯混合模型"""
    gmm = GaussianMixture(n_components=n_clusters, random_state=42,
                         covariance_type='full', n_init=5)
    labels = gmm.fit_predict(features)
    return labels, gmm

# 对不同特征应用不同聚类算法
print("\n应用聚类算法...")
results = {}

# 使用深度特征
print("\n--- 使用深度特征(ResNet18 + PCA) ---")
results['KMeans_Deep'] = apply_kmeans(deep_features_pca, n_clusters)
results['Spectral_Deep'] = apply_spectral(deep_features_pca, n_clusters)
results['Agglomerative_Deep'] = apply_agglomerative(deep_features_pca, n_clusters)
results['GMM_Deep'] = apply_gmm(deep_features_pca, n_clusters)

# 使用融合特征
print("\n--- 使用融合特征(深度+颜色+HOG) ---")
results['KMeans_Fusion'] = apply_kmeans(fusion_features, n_clusters)
results['Spectral_Fusion'] = apply_spectral(fusion_features, n_clusters)
results['Agglomerative_Fusion'] = apply_agglomerative(fusion_features, n_clusters)
results['GMM_Fusion'] = apply_gmm(fusion_features, n_clusters)

# ============================================================================
# 1.3 聚类效果评估 (5% Score)
# ============================================================================
"""
聚类评估指标说明：

无监督指标（不需要真实标签）：
1. Silhouette Score (轮廓系数): 衡量簇内紧密度和簇间分离度，范围[-1,1]，越大越好
2. Calinski-Harabasz Index: 簇间方差与簇内方差的比率，越大越好
3. Davies-Bouldin Index: 簇内散度与簇间距离的比率，越小越好

有监督指标（需要真实标签，用于验证）：
1. Normalized Mutual Information (NMI): 衡量聚类结果与真实标签的互信息，范围[0,1]
2. Adjusted Rand Index (ARI): 衡量聚类结果与真实标签的一致性，范围[-1,1]
3. Accuracy (通过匈牙利算法匹配): 将预测簇与真实标签最优匹配后的准确率
"""

print("\n" + "="*60)
print("1.3 聚类效果评估")
print("="*60)

from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                            davies_bouldin_score, normalized_mutual_info_score,
                            adjusted_rand_score, confusion_matrix)
from scipy.optimize import linear_sum_assignment

def cluster_accuracy(y_true, y_pred):
    """计算聚类准确率（使用匈牙利算法进行最优匹配）"""
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / len(y_true)

def evaluate_clustering(features, pred_labels, true_labels, method_name):
    """评估聚类效果"""
    # 无监督指标
    silhouette = silhouette_score(features, pred_labels)
    calinski = calinski_harabasz_score(features, pred_labels)
    davies = davies_bouldin_score(features, pred_labels)

    # 有监督指标（用于验证）
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    acc = cluster_accuracy(true_labels, pred_labels)

    return {
        'Method': method_name,
        'Silhouette': silhouette,
        'Calinski-Harabasz': calinski,
        'Davies-Bouldin': davies,
        'NMI': nmi,
        'ARI': ari,
        'Accuracy': acc
    }

# 评估所有聚类结果
print("\n聚类评估结果:")
print("-" * 100)
print(f"{'方法':<25} {'轮廓系数':>10} {'CH指数':>12} {'DB指数':>10} {'NMI':>8} {'ARI':>8} {'准确率':>8}")
print("-" * 100)

all_metrics = []

# 评估深度特征的聚类结果
for method in ['KMeans_Deep', 'Spectral_Deep', 'Agglomerative_Deep', 'GMM_Deep']:
    pred_labels = results[method][0]
    metrics = evaluate_clustering(deep_features_pca, pred_labels,
                                 true_labels_numeric, method)
    all_metrics.append(metrics)
    print(f"{metrics['Method']:<25} {metrics['Silhouette']:>10.4f} {metrics['Calinski-Harabasz']:>12.2f} "
          f"{metrics['Davies-Bouldin']:>10.4f} {metrics['NMI']:>8.4f} {metrics['ARI']:>8.4f} {metrics['Accuracy']:>8.4f}")

# 评估融合特征的聚类结果
for method in ['KMeans_Fusion', 'Spectral_Fusion', 'Agglomerative_Fusion', 'GMM_Fusion']:
    pred_labels = results[method][0]
    metrics = evaluate_clustering(fusion_features, pred_labels,
                                 true_labels_numeric, method)
    all_metrics.append(metrics)
    print(f"{metrics['Method']:<25} {metrics['Silhouette']:>10.4f} {metrics['Calinski-Harabasz']:>12.2f} "
          f"{metrics['Davies-Bouldin']:>10.4f} {metrics['NMI']:>8.4f} {metrics['ARI']:>8.4f} {metrics['Accuracy']:>8.4f}")

print("-" * 100)

# 找出最佳方法
best_by_nmi = max(all_metrics, key=lambda x: x['NMI'])
best_by_ari = max(all_metrics, key=lambda x: x['ARI'])
best_by_silhouette = max(all_metrics, key=lambda x: x['Silhouette'])

print(f"\n最佳方法（按NMI）: {best_by_nmi['Method']} (NMI={best_by_nmi['NMI']:.4f})")
print(f"最佳方法（按ARI）: {best_by_ari['Method']} (ARI={best_by_ari['ARI']:.4f})")
print(f"最佳方法（按轮廓系数）: {best_by_silhouette['Method']} (Silhouette={best_by_silhouette['Silhouette']:.4f})")

# ============================================================================
# 可视化
# ============================================================================
print("\n" + "="*60)
print("可视化结果")
print("="*60)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 使用t-SNE降维可视化
print("\n使用t-SNE进行可视化降维...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
features_2d = tsne.fit_transform(deep_features_pca)

# 创建可视化图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 真实标签
ax = axes[0, 0]
scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                     c=true_labels_numeric, cmap='tab10', alpha=0.7, s=20)
ax.set_title('True Labels', fontsize=12)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')

# K-Means结果
ax = axes[0, 1]
ax.scatter(features_2d[:, 0], features_2d[:, 1],
          c=results['KMeans_Deep'][0], cmap='tab10', alpha=0.7, s=20)
ax.set_title(f'K-Means (NMI={all_metrics[0]["NMI"]:.3f})', fontsize=12)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')

# Spectral结果
ax = axes[0, 2]
ax.scatter(features_2d[:, 0], features_2d[:, 1],
          c=results['Spectral_Deep'][0], cmap='tab10', alpha=0.7, s=20)
ax.set_title(f'Spectral (NMI={all_metrics[1]["NMI"]:.3f})', fontsize=12)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')

# Agglomerative结果
ax = axes[1, 0]
ax.scatter(features_2d[:, 0], features_2d[:, 1],
          c=results['Agglomerative_Deep'][0], cmap='tab10', alpha=0.7, s=20)
ax.set_title(f'Agglomerative (NMI={all_metrics[2]["NMI"]:.3f})', fontsize=12)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')

# GMM结果
ax = axes[1, 1]
ax.scatter(features_2d[:, 0], features_2d[:, 1],
          c=results['GMM_Deep'][0], cmap='tab10', alpha=0.7, s=20)
ax.set_title(f'GMM (NMI={all_metrics[3]["NMI"]:.3f})', fontsize=12)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')

# 评估指标对比
ax = axes[1, 2]
methods = [m['Method'].replace('_Deep', '') for m in all_metrics[:4]]
nmi_scores = [m['NMI'] for m in all_metrics[:4]]
ari_scores = [m['ARI'] for m in all_metrics[:4]]
x = np.arange(len(methods))
width = 0.35
bars1 = ax.bar(x - width/2, nmi_scores, width, label='NMI', color='steelblue')
bars2 = ax.bar(x + width/2, ari_scores, width, label='ARI', color='coral')
ax.set_ylabel('Score')
ax.set_title('Clustering Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45)
ax.legend()
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'clustering_results.png'), dpi=150, bbox_inches='tight')
print(f"\n可视化结果已保存到 {os.path.join(SCRIPT_DIR, 'clustering_results.png')}")
# plt.show()  # 注释掉避免阻塞，图像已保存到文件

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*60)
print("总结")
print("="*60)
print("""
1.1 图像特征处理方法:
    - 主要方法：使用预训练ResNet18提取512维深度特征
    - 辅助方法：颜色直方图(96维) + HOG特征(用于纹理)
    - 使用PCA降维到50维，保留主要信息同时减少噪声
    - 特征融合：将深度特征与传统特征结合

1.2 聚类算法选择:
    - K-Means: 简单高效，适合球形簇
    - Spectral Clustering: 适合非凸形状的簇
    - Agglomerative Clustering: 层次聚类，可发现数据结构
    - GMM: 概率模型，可处理不同形状的簇

1.3 评估指标:
    - 无监督指标: 轮廓系数、Calinski-Harabasz、Davies-Bouldin
    - 有监督指标: NMI、ARI、Accuracy（用于验证）
""")

# 保存最佳聚类结果
best_method = best_by_nmi['Method']
best_labels = results[best_method][0]

# 创建标签映射
from scipy.optimize import linear_sum_assignment
cm = confusion_matrix(true_labels_numeric, best_labels)
row_ind, col_ind = linear_sum_assignment(-cm)
label_mapping = {col: row for row, col in zip(row_ind, col_ind)}

# 保存结果
output_results = {}
for i, img_file in enumerate(image_files):
    output_results[img_file] = {
        'predicted_cluster': int(best_labels[i]),
        'true_label': true_labels[i]
    }

with open(os.path.join(SCRIPT_DIR, 'clustering_output.json'), 'w') as f:
    json.dump(output_results, f, indent=2)
print(f"\n聚类结果已保存到 {os.path.join(SCRIPT_DIR, 'clustering_output.json')}")
