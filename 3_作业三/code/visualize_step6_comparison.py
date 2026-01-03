"""
生成步骤4和步骤6的详细对比图
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 性能数据
steps = ['步骤4\n双向LSTM', '步骤6\n双向LSTM+注意力']
mae = [8.70, 9.07]
rmse = [14.84, 15.12]
r2 = [0.5688, 0.5522]

# 创建对比图
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('步骤4 vs 步骤6：注意力机制效果对比', fontsize=16, fontweight='bold', y=1.02)

colors = ['#4D96FF', '#FF6B6B']

# MAE对比
ax1 = axes[0]
bars1 = ax1.bar(steps, mae, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylabel('MAE (°C)', fontsize=13, fontweight='bold')
ax1.set_title('平均绝对误差 (越低越好)', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for bar, val in zip(bars1, mae):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}°C',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
    # 标注变化
    if val == mae[1]:
        change = ((mae[1] - mae[0]) / mae[0] * 100)
        ax1.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                f'{change:+.1f}%',
                ha='center', va='center', fontsize=11, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
ax1.set_ylim([0, max(mae) * 1.25])

# RMSE对比
ax2 = axes[1]
bars2 = ax2.bar(steps, rmse, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax2.set_ylabel('RMSE (°C)', fontsize=13, fontweight='bold')
ax2.set_title('均方根误差 (越低越好)', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
for bar, val in zip(bars2, rmse):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}°C',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
    if val == rmse[1]:
        change = ((rmse[1] - rmse[0]) / rmse[0] * 100)
        ax2.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                f'{change:+.1f}%',
                ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
ax2.set_ylim([0, max(rmse) * 1.25])

# R²对比
ax3 = axes[2]
bars3 = ax3.bar(steps, r2, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax3.set_ylabel('R² Score', fontsize=13, fontweight='bold')
ax3.set_title('决定系数 (越高越好)', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.axhline(y=0.6, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='目标: 0.6')
for bar, val in zip(bars3, r2):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.4f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
    if val == r2[1]:
        change = ((r2[1] - r2[0]) / r2[0] * 100)
        ax3.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                f'{change:+.1f}%',
                ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
ax3.set_ylim([0, 0.7])
ax3.legend(fontsize=10)

plt.tight_layout()
plt.savefig(r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\results\step4_vs_step6_comparison.png', 
            dpi=300, bbox_inches='tight')
print("对比图已保存至: results/step4_vs_step6_comparison.png")

# 创建注意力权重分析图
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
fig2.suptitle('步骤6：注意力权重分析', fontsize=16, fontweight='bold')

# 左图：注意力权重分布
time_steps = list(range(12))
attention_weights = [0.08474471, 0.0793976, 0.076221, 0.07558182, 0.07604345, 0.07693766,
                    0.07807862, 0.07953916, 0.08158267, 0.08436547, 0.09091952, 0.11658812]

bars = ax1.bar(time_steps, attention_weights, color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('时间步（10分钟/步）', fontsize=12, fontweight='bold')
ax1.set_ylabel('平均注意力权重', fontsize=12, fontweight='bold')
ax1.set_title('所有样本的平均注意力权重分布', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=1/12, color='red', linestyle='--', linewidth=2, alpha=0.7, label='均匀分布 (1/12)')

# 标注最高权重
max_idx = np.argmax(attention_weights)
max_weight = attention_weights[max_idx]
ax1.annotate(f'最高权重\n{max_weight:.4f}', 
           xy=(max_idx, max_weight),
           xytext=(max_idx-1, max_weight * 1.15),
           ha='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
           arrowprops=dict(arrowstyle='->', lw=2, color='red'))

ax1.legend(fontsize=10)
ax1.set_xticks(time_steps)

# 右图：累积注意力权重
cumulative_weights = np.cumsum(attention_weights)
ax2.plot(time_steps, cumulative_weights, marker='o', linewidth=2.5, markersize=8, 
         color='darkblue', label='累积注意力')
ax2.fill_between(time_steps, cumulative_weights, alpha=0.3, color='steelblue')
ax2.set_xlabel('时间步', fontsize=12, fontweight='bold')
ax2.set_ylabel('累积注意力权重', fontsize=12, fontweight='bold')
ax2.set_title('累积注意力权重曲线', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='50%阈值')
ax2.axhline(y=0.8, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='80%阈值')

# 标注50%和80%阈值对应的时间步
idx_50 = np.argmax(cumulative_weights >= 0.5)
idx_80 = np.argmax(cumulative_weights >= 0.8)
ax2.scatter([idx_50, idx_80], [cumulative_weights[idx_50], cumulative_weights[idx_80]], 
           s=200, c='red', zorder=5, marker='*')
ax2.text(idx_50, cumulative_weights[idx_50] + 0.05, f'50%@步骤{idx_50}', 
        ha='center', fontsize=10, fontweight='bold')
ax2.text(idx_80, cumulative_weights[idx_80] + 0.05, f'80%@步骤{idx_80}', 
        ha='center', fontsize=10, fontweight='bold')

ax2.legend(fontsize=10)
ax2.set_xticks(time_steps)
ax2.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig(r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\results\step6_attention_analysis.png', 
            dpi=300, bbox_inches='tight')
print("注意力分析图已保存至: results/step6_attention_analysis.png")

# 创建结论总结图
fig3, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# 标题
title_text = "步骤6实验总结：注意力机制未能提升性能"
ax.text(0.5, 0.95, title_text, fontsize=20, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))

# 性能对比
perf_text = """
性能对比：

┌─────────────┬──────────┬──────────┬──────────┐
│   指标      │  步骤4   │  步骤6   │  变化    │
├─────────────┼──────────┼──────────┼──────────┤
│ MAE (°C)    │   8.70   │   9.07   │  +4.3%  │
│ RMSE (°C)   │  14.84   │  15.12   │  +1.9%  │
│ R²          │  0.5688  │  0.5522  │  -2.9%  │
└─────────────┴──────────┴──────────┴──────────┘

✅ 步骤4（不用注意力）表现更好
"""
ax.text(0.5, 0.75, perf_text, fontsize=11, ha='center', va='top',
        family='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))

# 原因分析
reason_text = """
为什么注意力没有帮助？

1. 注意力分布过于分散（熵=2.45，接近均匀）
   → 没有识别出关键时间点

2. 最后时间步权重最高（0.117）
   → 证明最近时刻最重要
   → 步骤4直接用最后时间步是正确的

3. 双向LSTM已经整合了序列信息
   → 注意力的作用被弱化

4. 温度预测任务特点：强自相关、短期依赖
   → 不需要复杂的注意力机制
"""
ax.text(0.5, 0.45, reason_text, fontsize=10, ha='center', va='top',
        bbox=dict(boxstyle='round,pad=1', facecolor='#FFE5CC', alpha=0.8))

# 建议
suggestion_text = """
改进建议：

✅ 保持步骤4作为最佳模型（R²=0.57, MAE=8.70）
❌ 放弃当前的注意力机制（未带来提升）
🎯 尝试更有前景的方向：
   • 特征工程（添加移动统计、趋势特征）⭐⭐⭐⭐⭐
   • 集成学习（训练多个模型取平均）⭐⭐⭐⭐
   • 调整序列长度（尝试3-4小时窗口）⭐⭐⭐⭐
"""
ax.text(0.5, 0.15, suggestion_text, fontsize=10, ha='center', va='top',
        bbox=dict(boxstyle='round,pad=1', facecolor='#CCFFCC', alpha=0.8))

plt.tight_layout()
plt.savefig(r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\results\step6_conclusion_summary.png', 
            dpi=300, bbox_inches='tight')
print("总结图已保存至: results/step6_conclusion_summary.png")

print("\n所有对比图表生成完成！")
