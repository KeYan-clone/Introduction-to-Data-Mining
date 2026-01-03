"""
步骤7分析可视化脚本

生成详细的对比图表，展示特征工程的效果
"""

import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

results_dir = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\results'

# 各步骤的性能数据
steps = ['步骤1\n基线', '步骤2\nLSTM', '步骤3\nLSTM+时间', '步骤4\nBiLSTM+时间', '步骤6\n注意力机制', '步骤7\n特征工程']
mae_values = [13.38, 9.47, 8.98, 8.70, 9.07, 3.65]
rmse_values = [23.04, 17.75, 15.65, 14.84, 15.12, 5.94]
r2_values = [0.1287, 0.4466, 0.5284, 0.5688, 0.5522, 0.9310]

# ==================== 图1: 所有步骤的MAE对比 ====================
fig1, ax1 = plt.subplots(figsize=(12, 6))

colors = ['#95a5a6', '#3498db', '#9b59b6', '#2ecc71', '#e74c3c', '#f39c12']
bars = ax1.bar(steps, mae_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 在柱子上添加数值
for i, (bar, val) in enumerate(zip(bars, mae_values)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}°C',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加改进百分比标注
for i in range(1, len(mae_values)):
    improvement = (mae_values[i-1] - mae_values[i]) / mae_values[i-1] * 100
    if improvement > 0:
        ax1.annotate(f'↓{improvement:.1f}%',
                    xy=(i-0.5, (mae_values[i-1] + mae_values[i])/2),
                    xytext=(i-0.2, (mae_values[i-1] + mae_values[i])/2),
                    fontsize=9, color='green', fontweight='bold')

ax1.set_ylabel('平均绝对误差 MAE (°C)', fontsize=12, fontweight='bold')
ax1.set_title('所有实验步骤的MAE对比（越低越好）', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(mae_values) * 1.2)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'step7_mae_comparison.png'), dpi=300, bbox_inches='tight')
print("✅ 已生成: step7_mae_comparison.png")
plt.close()

# ==================== 图2: 所有步骤的R²对比 ====================
fig2, ax2 = plt.subplots(figsize=(12, 6))

bars = ax2.bar(steps, r2_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 在柱子上添加数值
for bar, val in zip(bars, r2_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加改进百分比标注
for i in range(1, len(r2_values)):
    improvement = (r2_values[i] - r2_values[i-1]) / r2_values[i-1] * 100
    if improvement > 0:
        ax2.annotate(f'↑{improvement:.1f}%',
                    xy=(i-0.5, (r2_values[i-1] + r2_values[i])/2),
                    xytext=(i-0.2, (r2_values[i-1] + r2_values[i])/2),
                    fontsize=9, color='green', fontweight='bold')

ax2.set_ylabel('决定系数 R² Score', fontsize=12, fontweight='bold')
ax2.set_title('所有实验步骤的R²对比（越高越好）', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, 1.0)

# 添加参考线
ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='优秀阈值 (0.9)')
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'step7_r2_comparison.png'), dpi=300, bbox_inches='tight')
print("✅ 已生成: step7_r2_comparison.png")
plt.close()

# ==================== 图3: 步骤4 vs 步骤7 详细对比 ====================
fig3, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['MAE (°C)', 'RMSE (°C)', 'R² Score']
step4_values = [8.70, 14.84, 0.5688]
step7_values = [3.65, 5.94, 0.9310]

for idx, (ax, metric, s4, s7) in enumerate(zip(axes, metrics, step4_values, step7_values)):
    bars = ax.bar(['步骤4\nBiLSTM+时间', '步骤7\n特征工程'], [s4, s7], 
                   color=['#2ecc71', '#f39c12'], alpha=0.8, edgecolor='black', linewidth=2)
    
    # 添加数值标签
    for bar, val in zip(bars, [s4, s7]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}' if idx < 2 else f'{val:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 计算改进
    if idx < 2:  # MAE和RMSE越小越好
        improvement = (s4 - s7) / s4 * 100
        ax.text(0.5, max(s4, s7) * 0.5, 
                f'↓ {improvement:.1f}%\n改进',
                ha='center', fontsize=14, fontweight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    else:  # R²越大越好
        improvement = (s7 - s4) / s4 * 100
        ax.text(0.5, (s4 + s7) * 0.5,
                f'↑ {improvement:.1f}%\n提升',
                ha='center', fontsize=14, fontweight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(f'{metric}对比', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

fig3.suptitle('步骤7特征工程 vs 步骤4基线 - 详细性能对比', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'step7_vs_step4_detailed.png'), dpi=300, bbox_inches='tight')
print("✅ 已生成: step7_vs_step4_detailed.png")
plt.close()

# ==================== 图4: 特征工程的增量效果分析 ====================
fig4, ax4 = plt.subplots(figsize=(12, 7))

# 创建增量改进图
categories = ['步骤1→2\n引入LSTM', '步骤2→3\n时间特征', '步骤3→4\n双向LSTM', 
              '步骤4→6\n注意力机制', '步骤4→7\n特征工程']
mae_improvements = [
    (13.38 - 9.47) / 13.38 * 100,   # 步骤1→2
    (9.47 - 8.98) / 9.47 * 100,      # 步骤2→3
    (8.98 - 8.70) / 8.98 * 100,      # 步骤3→4
    (8.70 - 9.07) / 8.70 * 100,      # 步骤4→6 (负值)
    (8.70 - 3.65) / 8.70 * 100       # 步骤4→7
]

colors_improve = ['green' if x > 0 else 'red' for x in mae_improvements]
bars = ax4.barh(categories, mae_improvements, color=colors_improve, alpha=0.7, edgecolor='black', linewidth=1.5)

# 添加数值标签
for bar, val in zip(bars, mae_improvements):
    width = bar.get_width()
    label_x = width + (1 if width > 0 else -1)
    ax4.text(label_x, bar.get_y() + bar.get_height()/2.,
             f'{val:+.1f}%',
             ha='left' if val > 0 else 'right', va='center', 
             fontsize=12, fontweight='bold')

ax4.axvline(x=0, color='black', linewidth=1)
ax4.set_xlabel('MAE改进百分比 (%)', fontsize=12, fontweight='bold')
ax4.set_title('各实验改进的增量效果分析（正值=改进，负值=退步）', fontsize=14, fontweight='bold')
ax4.grid(axis='x', alpha=0.3, linestyle='--')

# 添加注释
ax4.text(55, 0.5, '⭐ 特征工程带来\n最显著的改进！', 
         fontsize=11, fontweight='bold', color='darkgreen',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'step7_incremental_improvements.png'), dpi=300, bbox_inches='tight')
print("✅ 已生成: step7_incremental_improvements.png")
plt.close()

# ==================== 图5: 综合评分雷达图 ====================
from math import pi

fig5, ax5 = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# 准备数据（归一化到0-1）
categories_radar = ['MAE\n(越低越好)', 'RMSE\n(越低越好)', 'R²\n(越高越好)', 
                    '训练时间\n(越短越好)', '模型复杂度\n(越简单越好)']
N = len(categories_radar)

# 归一化分数（0-1，1最好）
step4_scores = [
    1 - 8.70/13.38,      # MAE归一化
    1 - 14.84/23.04,     # RMSE归一化
    0.5688,              # R²本身就是0-1
    0.8,                 # 训练时间（主观评分）
    0.9                  # 模型复杂度（主观评分）
]

step7_scores = [
    1 - 3.65/13.38,      # MAE归一化
    1 - 5.94/23.04,      # RMSE归一化
    0.9310,              # R²
    0.7,                 # 训练时间稍长
    0.7                  # 复杂度增加
]

# 计算角度
angles = [n / float(N) * 2 * pi for n in range(N)]
step4_scores += step4_scores[:1]
step7_scores += step7_scores[:1]
angles += angles[:1]

# 绘制
ax5.plot(angles, step4_scores, 'o-', linewidth=2, label='步骤4 BiLSTM+时间', color='#2ecc71')
ax5.fill(angles, step4_scores, alpha=0.25, color='#2ecc71')

ax5.plot(angles, step7_scores, 'o-', linewidth=2, label='步骤7 特征工程', color='#f39c12')
ax5.fill(angles, step7_scores, alpha=0.25, color='#f39c12')

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(categories_radar, fontsize=11)
ax5.set_ylim(0, 1)
ax5.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax5.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax5.grid(True)

ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
ax5.set_title('步骤4 vs 步骤7 综合评分对比', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'step7_radar_comparison.png'), dpi=300, bbox_inches='tight')
print("✅ 已生成: step7_radar_comparison.png")
plt.close()

print("\n" + "="*50)
print("所有分析图表生成完成！")
print("="*50)
print("\n生成的图表文件：")
print("1. step7_mae_comparison.png - 所有步骤MAE对比")
print("2. step7_r2_comparison.png - 所有步骤R²对比")
print("3. step7_vs_step4_detailed.png - 步骤4 vs 7详细对比")
print("4. step7_incremental_improvements.png - 增量改进分析")
print("5. step7_radar_comparison.png - 综合评分雷达图")
