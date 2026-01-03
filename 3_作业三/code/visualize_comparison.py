"""
生成所有步骤的性能对比图表
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 各步骤的性能指标
steps = ['步骤1\n基线', '步骤2\nLSTM', '步骤3\nLSTM+时间', '步骤4\n双向LSTM', '步骤5\n双向LSTM+Delta']
mse = [237.84, 348.48, 242.61, 220.23, 13.58]
rmse = [15.42, 18.67, 15.58, 14.84, 3.68]
mae = [9.58, 11.43, 9.24, 8.70, 2.11]
r2 = [0.5343, 0.3177, 0.5250, 0.5688, 0.9734]

# 创建2x2子图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('时序预测模型性能对比：步骤1-5', fontsize=18, fontweight='bold', y=0.995)

colors = ['#FF6B6B', '#FFA07A', '#FFD93D', '#6BCB77', '#4D96FF']

# MSE对比
ax1 = axes[0, 0]
bars1 = ax1.bar(steps, mse, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('MSE', fontsize=13, fontweight='bold')
ax1.set_title('均方误差（MSE）- 越低越好', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, val) in enumerate(zip(bars1, mse)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.set_ylim([0, max(mse) * 1.15])

# RMSE对比
ax2 = axes[0, 1]
bars2 = ax2.bar(steps, rmse, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('RMSE (°C)', fontsize=13, fontweight='bold')
ax2.set_title('均方根误差（RMSE）- 越低越好', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, val) in enumerate(zip(bars2, rmse)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.set_ylim([0, max(rmse) * 1.15])

# MAE对比
ax3 = axes[1, 0]
bars3 = ax3.bar(steps, mae, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('MAE (°C)', fontsize=13, fontweight='bold')
ax3.set_title('平均绝对误差（MAE）- 越低越好', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, val) in enumerate(zip(bars3, mae)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
ax3.set_ylim([0, max(mae) * 1.15])

# R²对比
ax4 = axes[1, 1]
bars4 = ax4.bar(steps, r2, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('R² Score', fontsize=13, fontweight='bold')
ax4.set_title('决定系数（R²）- 越高越好', fontsize=14, fontweight='bold')
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='完美预测')
for i, (bar, val) in enumerate(zip(bars4, r2)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
ax4.set_ylim([0, 1.1])
ax4.legend(fontsize=10)

plt.tight_layout()
plt.savefig(r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\results\all_steps_comparison.png', 
            dpi=300, bbox_inches='tight')
print("对比图表已保存至: results/all_steps_comparison.png")

# 创建改进百分比图
fig2, ax = plt.subplots(figsize=(12, 7))

metrics_names = ['MSE降低', 'RMSE降低', 'MAE降低', 'R²提升']
step4_vals = [220.23, 14.84, 8.70, 0.5688]
step5_vals = [13.58, 3.68, 2.11, 0.9734]

improvements = []
for i in range(3):
    improvement = (step4_vals[i] - step5_vals[i]) / step4_vals[i] * 100
    improvements.append(improvement)
# R²是越大越好，所以计算方式不同
r2_improvement = (step5_vals[3] - step4_vals[3]) / step4_vals[3] * 100
improvements.append(r2_improvement)

bars = ax.barh(metrics_names, improvements, color=['#4D96FF', '#4D96FF', '#4D96FF', '#6BCB77'], 
               alpha=0.8, edgecolor='black', linewidth=2)
ax.set_xlabel('改进百分比 (%)', fontsize=13, fontweight='bold')
ax.set_title('步骤5相对步骤4的性能改进', fontsize=16, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')

for i, (bar, val) in enumerate(zip(bars, improvements)):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{val:.1f}%',
            ha='left', va='center', fontsize=13, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

ax.set_xlim([0, max(improvements) * 1.2])

plt.tight_layout()
plt.savefig(r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\results\step5_improvement_percentage.png', 
            dpi=300, bbox_inches='tight')
print("改进百分比图已保存至: results/step5_improvement_percentage.png")

plt.show()
