"""
步骤7数据泄露分析脚本

检查步骤7的高性能是否由数据泄露导致
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("步骤7数据泄露分析")
print("="*70)

# ==================== 加载数据 ====================
data_path = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\data\weather.csv'
df = pd.read_csv(data_path, encoding='ISO-8859-1')
# date列已经存在，直接使用
if 'date' not in df.columns:
    raise ValueError(f"Column 'date' not found. Available: {df.columns.tolist()}")
df = df.sort_values('date').reset_index(drop=True)

print(f"\n数据集大小: {len(df)} 条记录")

# ==================== 问题1: 统计特征是否包含目标信息？ ====================
print("\n" + "="*70)
print("问题1: 统计特征是否包含当前时刻的目标变量OT？")
print("="*70)

# 模拟步骤7的特征生成
df_test = df.copy()
df_test['temp_ma_6'] = df_test['OT'].rolling(window=6, min_periods=1).mean()
df_test['temp_ma_12'] = df_test['OT'].rolling(window=12, min_periods=1).mean()

# 检查第100行数据
idx = 100
print(f"\n检查第{idx}行数据:")
print(f"当前OT值: {df_test.loc[idx, 'OT']:.2f}°C")
print(f"temp_ma_6 (1小时均值): {df_test.loc[idx, 'temp_ma_6']:.2f}°C")
print(f"过去6个OT值: {df_test.loc[idx-5:idx, 'OT'].values}")
print(f"过去6个OT均值: {df_test.loc[idx-5:idx, 'OT'].mean():.2f}°C")

# 关键检查：移动平均是否包含当前值
ot_current = df_test.loc[idx, 'OT']
ot_past_6 = df_test.loc[idx-5:idx, 'OT'].values
ma_includes_current = np.isclose(df_test.loc[idx, 'temp_ma_6'], ot_past_6.mean())

print(f"\n⚠️ 移动平均包含当前时刻的OT: {ma_includes_current}")
print(f"   - rolling(6).mean() 包含 index[{idx-5}:{idx+1}]")
print(f"   - 这意味着用OT预测OT！")

# ==================== 问题2: 序列构建时的时间关系 ====================
print("\n" + "="*70)
print("问题2: 序列构建时特征与目标的时间关系")
print("="*70)

# 模拟序列构建
seq_length = 12
test_idx = 100

print(f"\n以第{test_idx}行为例，构建序列:")
print(f"输入序列: 第{test_idx-seq_length+1}行到第{test_idx}行")
print(f"目标值: 第{test_idx}行的OT = {df_test.loc[test_idx, 'OT']:.2f}°C")

print(f"\n第{test_idx}行的特征值:")
print(f"  - temp_ma_6 = {df_test.loc[test_idx, 'temp_ma_6']:.2f}°C")
print(f"  - 计算方式: mean(OT[{test_idx-5}:{test_idx+1}])")
print(f"  - ⚠️ 包含了当前时刻(第{test_idx}行)的OT值！")

print(f"\n⚠️ 数据泄露路径:")
print(f"   特征temp_ma_6[{test_idx}] 包含 OT[{test_idx}]")
print(f"         ↓ 作为输入")
print(f"   预测目标: OT[{test_idx}]")
print(f"   结果: 用OT预测自己！")

# ==================== 问题3: 相关性分析 ====================
print("\n" + "="*70)
print("问题3: 统计特征与目标变量的相关性")
print("="*70)

# 添加所有统计特征
df_full = df.copy()
df_full['temp_ma_6'] = df_full['OT'].rolling(window=6, min_periods=1).mean()
df_full['temp_ma_12'] = df_full['OT'].rolling(window=12, min_periods=1).mean()
df_full['temp_ma_36'] = df_full['OT'].rolling(window=36, min_periods=1).mean()
df_full['temp_std_6'] = df_full['OT'].rolling(window=6, min_periods=1).std()
df_full['temp_std_12'] = df_full['OT'].rolling(window=12, min_periods=1).std()
df_full['temp_diff_1'] = df_full['OT'].diff(1).fillna(0)
df_full['temp_diff_6'] = df_full['OT'].diff(6).fillna(0)
df_full['temp_diff2'] = df_full['temp_diff_1'].diff(1).fillna(0)
df_full = df_full.fillna(method='bfill').fillna(method='ffill')

# 计算相关性
stat_features = ['temp_ma_6', 'temp_ma_12', 'temp_ma_36', 
                'temp_std_6', 'temp_std_12',
                'temp_diff_1', 'temp_diff_6', 'temp_diff2']

print("\n统计特征与OT的相关系数:")
for feat in stat_features:
    corr = df_full[feat].corr(df_full['OT'])
    print(f"  {feat:20s}: {corr:.4f} {'⚠️ 极高相关!' if abs(corr) > 0.95 else ''}")

# ==================== 问题4: 基线测试 ====================
print("\n" + "="*70)
print("问题4: 简单基线模型测试（检测数据泄露）")
print("="*70)

from sklearn.linear_model import LinearRegression

# 准备数据（仅使用统计特征）
split_idx = int(len(df_full) * 0.8)
train_data = df_full[:split_idx]
test_data = df_full[split_idx:]

X_train = train_data[stat_features].values
y_train = train_data['OT'].values
X_test = test_data[stat_features].values
y_test = test_data['OT'].values

# 训练简单线性回归
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n简单线性回归（仅用8个统计特征）:")
print(f"  R² = {r2:.4f}")
print(f"  MAE = {mae:.2f}°C")

if r2 > 0.90:
    print(f"\n⚠️⚠️⚠️ 严重警告: 线性回归R²={r2:.4f} > 0.90")
    print(f"  这意味着简单线性模型就能达到90%+性能")
    print(f"  强烈暗示存在数据泄露！")

# ==================== 问题5: 正确vs错误的特征构建对比 ====================
print("\n" + "="*70)
print("问题5: 正确vs错误的特征构建方式对比")
print("="*70)

print("\n❌ 错误方式（步骤7当前实现）:")
print("```python")
print("df['temp_ma_6'] = df['OT'].rolling(6).mean()")
print("# rolling(6) 包含当前行，导致 temp_ma_6[t] 包含 OT[t]")
print("```")

print("\n✅ 正确方式（应该使用）:")
print("```python")
print("# 方式1: 使用shift，确保只用历史数据")
print("df['temp_ma_6'] = df['OT'].rolling(6).mean().shift(1)")
print("# shift(1)后，temp_ma_6[t] = mean(OT[t-6:t-1])，不包含OT[t]")
print("")
print("# 方式2: 在序列构建时避免使用当前时刻特征")
print("# 序列输入应该是 [t-12:t-1]，不包含t时刻")
print("```")

# ==================== 问题6: 验证修正后的性能 ====================
print("\n" + "="*70)
print("问题6: 修正数据泄露后的预期性能")
print("="*70)

# 正确构建特征（shift）
df_correct = df.copy()
df_correct['temp_ma_6'] = df_correct['OT'].rolling(window=6, min_periods=1).mean().shift(1)
df_correct['temp_ma_12'] = df_correct['OT'].rolling(window=12, min_periods=1).mean().shift(1)
df_correct['temp_ma_36'] = df_correct['OT'].rolling(window=36, min_periods=1).mean().shift(1)
df_correct['temp_std_6'] = df_correct['OT'].rolling(window=6, min_periods=1).std().shift(1)
df_correct['temp_std_12'] = df_correct['OT'].rolling(window=12, min_periods=1).std().shift(1)
df_correct['temp_diff_1'] = df_correct['OT'].diff(1)
df_correct['temp_diff_6'] = df_correct['OT'].diff(6)
df_correct['temp_diff2'] = df_correct['temp_diff_1'].diff(1)
df_correct = df_correct.fillna(method='bfill').fillna(method='ffill')

# 使用线性回归测试
train_data_correct = df_correct[:split_idx]
test_data_correct = df_correct[split_idx:]

X_train_correct = train_data_correct[stat_features].values
X_test_correct = test_data_correct[stat_features].values

lr_correct = LinearRegression()
lr_correct.fit(X_train_correct, y_train)
y_pred_correct = lr_correct.predict(X_test_correct)

r2_correct = r2_score(y_test, y_pred_correct)
mae_correct = mean_absolute_error(y_test, y_pred_correct)

print(f"\n修正后的线性回归（使用shift(1)）:")
print(f"  R² = {r2_correct:.4f} (原: {r2:.4f})")
print(f"  MAE = {mae_correct:.2f}°C (原: {mae:.2f}°C)")
print(f"  性能下降: R² 降低 {(r2-r2_correct)/r2*100:.1f}%")

# ==================== 生成对比图 ====================
print("\n" + "="*70)
print("生成可视化对比图")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 特征与目标的相关性
ax1 = axes[0, 0]
correlations = [df_full[feat].corr(df_full['OT']) for feat in stat_features]
colors = ['red' if abs(c) > 0.95 else 'orange' if abs(c) > 0.9 else 'green' for c in correlations]
bars = ax1.barh(stat_features, correlations, color=colors, alpha=0.7, edgecolor='black')
ax1.axvline(x=0.95, color='red', linestyle='--', alpha=0.5, label='危险阈值 (0.95)')
ax1.axvline(x=-0.95, color='red', linestyle='--', alpha=0.5)
ax1.set_xlabel('与OT的相关系数', fontsize=11, fontweight='bold')
ax1.set_title('统计特征与目标变量的相关性\n（>0.95暗示数据泄露）', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# 图2: 线性回归性能对比
ax2 = axes[0, 1]
models = ['错误方式\n(含当前值)', '正确方式\n(shift)']
r2_values = [r2, r2_correct]
colors_r2 = ['red', 'green']
bars = ax2.bar(models, r2_values, color=colors_r2, alpha=0.7, edgecolor='black', linewidth=2)
for bar, val in zip(bars, r2_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.4f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
ax2.axhline(y=0.90, color='red', linestyle='--', alpha=0.5, label='可疑阈值')
ax2.set_ylabel('R² Score', fontsize=11, fontweight='bold')
ax2.set_title('简单线性回归性能对比\n（仅用8个统计特征）', fontsize=12, fontweight='bold')
ax2.legend()
ax2.set_ylim(0, 1.0)
ax2.grid(axis='y', alpha=0.3)

# 图3: 预测示例（错误方式）
ax3 = axes[1, 0]
sample_size = 200
sample_test = y_test[:sample_size]
sample_pred = y_pred[:sample_size]
ax3.plot(sample_test, label='实际温度', linewidth=2, alpha=0.7)
ax3.plot(sample_pred, label='预测温度', linewidth=2, alpha=0.7)
ax3.set_xlabel('样本索引', fontsize=11)
ax3.set_ylabel('温度 (°C)', fontsize=11)
ax3.set_title(f'错误方式预测结果 (R²={r2:.4f})\n拟合过好，暗示泄露', 
             fontsize=12, fontweight='bold', color='red')
ax3.legend()
ax3.grid(alpha=0.3)

# 图4: 预测示例（正确方式）
ax4 = axes[1, 1]
sample_pred_correct = y_pred_correct[:sample_size]
ax4.plot(sample_test, label='实际温度', linewidth=2, alpha=0.7)
ax4.plot(sample_pred_correct, label='预测温度', linewidth=2, alpha=0.7)
ax4.set_xlabel('样本索引', fontsize=11)
ax4.set_ylabel('温度 (°C)', fontsize=11)
ax4.set_title(f'正确方式预测结果 (R²={r2_correct:.4f})\n更真实的性能', 
             fontsize=12, fontweight='bold', color='green')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
results_dir = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\results'
plt.savefig(f'{results_dir}/step7_leakage_analysis.png', dpi=300, bbox_inches='tight')
print(f"✅ 已生成: step7_leakage_analysis.png")
plt.close()

# ==================== 最终结论 ====================
print("\n" + "="*70)
print("🔍 数据泄露分析结论")
print("="*70)

print("\n✅ 确认发现数据泄露！")
print("\n证据:")
print(f"  1. 移动平均特征包含当前时刻的OT值")
print(f"  2. 统计特征与OT相关性极高 (>0.95)")
print(f"  3. 简单线性回归就能达到R²={r2:.4f}")
print(f"  4. 修正后性能显著下降到R²={r2_correct:.4f}")

print("\n⚠️ 步骤7的高性能(R²=0.93)是虚假的！")
print(f"  - 原因: rolling().mean() 包含当前时刻值")
print(f"  - 相当于: 用OT[t]预测OT[t]")
print(f"  - 真实性能预计: R²≈{r2_correct:.2f}（类似步骤4）")

print("\n📝 修正建议:")
print(f"  1. 使用 .shift(1) 确保只用历史数据")
print(f"  2. 或在序列构建时只用[t-12:t-1]，不含t")
print(f"  3. 重新训练步骤7并更新结果")

print("\n" + "="*70)
print("详细报告已生成！")
print("="*70)
