"""
数据泄露检查脚本
分析步骤5是否存在数据泄露导致R²虚高的问题
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from utils import (set_seed, get_device, load_and_preprocess_data, 
                   extract_time_features, create_feature_target_sequences_with_delta, 
                   WeatherDataset)

set_seed(42)
device = get_device()

print("="*70)
print("数据泄露检查分析")
print("="*70)

# 加载数据
data_path = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\data\weather.csv'
df = load_and_preprocess_data(data_path)
df_with_time = extract_time_features(df)

feature_cols = [col for col in df.columns if col not in ['date', 'OT']]
time_feature_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
feature_cols_extended = feature_cols + time_feature_cols
features_extended = df_with_time[feature_cols_extended].values
target = df_with_time['OT'].values.reshape(-1, 1)

seq_length = 12
X, y_delta, y_last = create_feature_target_sequences_with_delta(features_extended, target, seq_length)

split_idx = int(len(X) * 0.8)
y_delta_test = y_delta[split_idx:]
y_last_test = y_last[split_idx:]

print("\n【检查1】特征是否包含目标变量（OT）")
print("-" * 70)
print(f"原始特征列: {feature_cols[:5]}... (共{len(feature_cols)}个)")
print(f"是否包含'OT': {'OT' in feature_cols}")
print("✅ 正确：输入特征不包含目标变量OT")

print("\n【检查2】Delta和y_last的定义")
print("-" * 70)
print(f"y_delta定义: OT(t+{seq_length}) - OT(t+{seq_length-1})")
print(f"y_last定义: OT(t+{seq_length-1})")
print(f"预测公式: OT(t+{seq_length}) = y_last + delta_pred")
print("\n关键问题：y_last是序列的最后一个点，但输入X中不包含历史OT值！")
print("这意味着模型无法从输入中直接获取历史OT信息。")

print("\n【检查3】Delta统计分析")
print("-" * 70)
actuals_absolute = y_last_test + y_delta_test
print(f"测试集Delta统计:")
print(f"  - Mean: {np.mean(y_delta_test):.4f}°C")
print(f"  - Std: {np.std(y_delta_test):.4f}°C")
print(f"  - Median: {np.median(y_delta_test):.4f}°C")
print(f"  - Min: {np.min(y_delta_test):.4f}°C, Max: {np.max(y_delta_test):.4f}°C")
print(f"\n如果模型总是预测delta=0（即温度不变）:")
baseline_mae_delta = np.mean(np.abs(y_delta_test))
print(f"  - Baseline MAE: {baseline_mae_delta:.4f}°C")

print("\n【检查4】加载训练好的模型进行分析")
print("-" * 70)

# 重新加载模型和数据
from step_5_temp import BiLSTMDeltaModel  # 临时导入

n_features = X.shape[2]
scaler_X = StandardScaler()
scaler_delta = StandardScaler()

X_train, X_test = X[:split_idx], X[split_idx:]
y_delta_train, y_delta_test = y_delta[:split_idx], y_delta[split_idx:]
y_last_train, y_last_test = y_last[:split_idx], y_last[split_idx:]

X_train_flat = X_train.reshape(-1, n_features)
X_test_flat = X_test.reshape(-1, n_features)
scaler_X.fit(X_train_flat)
X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test.shape)

scaler_delta.fit(y_delta_train.reshape(-1, 1))
y_delta_test_scaled = scaler_delta.transform(y_delta_test.reshape(-1, 1)).flatten()

# 创建模型并重新训练（快速版本）
input_size = X_train.shape[2]
model = BiLSTMDeltaModel(input_size).to(device)

X_train_scaled = scaler_X.transform(X_train_flat).reshape(X_train.shape)
y_delta_train_scaled = scaler_delta.transform(y_delta_train.reshape(-1, 1)).flatten()

train_dataset = WeatherDataset(X_train_scaled, y_delta_train_scaled, y_last_train)
test_dataset = WeatherDataset(X_test_scaled, y_delta_test_scaled, y_last_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("快速训练模型（20 epochs）...")
model.train()
for epoch in range(20):
    for batch in train_loader:
        X_batch, y_delta_batch, _ = batch
        X_batch, y_delta_batch = X_batch.to(device), y_delta_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_delta_batch)
        loss.backward()
        optimizer.step()

print("\n评估模型预测...")
model.eval()
predictions_delta = []
with torch.no_grad():
    for batch in test_loader:
        X_batch, _, _ = batch
        X_batch = X_batch.to(device)
        outputs = model(X_batch).squeeze()
        predictions_delta.extend(outputs.cpu().numpy())

predictions_delta = np.array(predictions_delta).reshape(-1, 1)
predictions_delta_original = scaler_delta.inverse_transform(predictions_delta).flatten()

print("\n【检查5】模型预测的Delta分布")
print("-" * 70)
print(f"预测Delta统计:")
print(f"  - Mean: {np.mean(predictions_delta_original):.4f}°C")
print(f"  - Std: {np.std(predictions_delta_original):.4f}°C")
print(f"  - Median: {np.median(predictions_delta_original):.4f}°C")
print(f"  - Min: {np.min(predictions_delta_original):.4f}°C, Max: {np.max(predictions_delta_original):.4f}°C")

print("\n【关键发现】预测Delta与真实Delta的差异:")
print(f"  - 真实Delta Std: {np.std(y_delta_test):.4f}°C")
print(f"  - 预测Delta Std: {np.std(predictions_delta_original):.4f}°C")
if np.std(predictions_delta_original) < 0.5:
    print("  ⚠️  警告：预测Delta的标准差过小，模型可能在预测接近0的值！")
else:
    print("  ✅ 预测Delta分布正常")

print("\n【检查6】不同评估方式的对比")
print("-" * 70)

# 方法1：使用真实y_last（当前实现）
predictions_abs_with_last = y_last_test + predictions_delta_original
actuals_absolute = y_last_test + y_delta_test

mae_with_last = mean_absolute_error(actuals_absolute, predictions_abs_with_last)
r2_with_last = r2_score(actuals_absolute, predictions_abs_with_last)

print(f"方法1 - 使用真实y_last（当前实现）:")
print(f"  MAE: {mae_with_last:.4f}°C")
print(f"  R²: {r2_with_last:.4f}")

# 方法2：仅评估Delta预测能力
mae_delta = mean_absolute_error(y_delta_test, predictions_delta_original)
r2_delta = r2_score(y_delta_test, predictions_delta_original)

print(f"\n方法2 - 仅评估Delta预测能力:")
print(f"  MAE: {mae_delta:.4f}°C")
print(f"  R²: {r2_delta:.4f}")

# 方法3：Baseline（预测delta=0）
predictions_delta_zero = np.zeros_like(y_delta_test)
predictions_abs_baseline = y_last_test + predictions_delta_zero
mae_baseline = mean_absolute_error(actuals_absolute, predictions_abs_baseline)
r2_baseline = r2_score(actuals_absolute, predictions_abs_baseline)

print(f"\n方法3 - Baseline（预测delta=0，即温度不变）:")
print(f"  MAE: {mae_baseline:.4f}°C")
print(f"  R²: {r2_baseline:.4f}")

print("\n【检查7】数据泄露判断")
print("-" * 70)

leakage_detected = False

# 判断1：预测delta是否过于接近0
if np.std(predictions_delta_original) < 0.5:
    print("⚠️  问题1: 模型预测的delta标准差过小（<0.5），几乎总是预测温度不变")
    leakage_detected = True

# 判断2：使用y_last是否合理
print("\n问题2: 使用y_last（上一时刻真实温度）是否构成数据泄露?")
print("  分析：")
print("    - y_last是序列最后一个点（t-1时刻）的真实温度")
print("    - 但输入特征X不包含历史OT值")
print("    - 在真实预测场景中，我们确实知道当前时刻的温度")
print("    - 关键：模型是否能从其他特征推断出y_last？")

# 检查：是否可以从其他特征重构OT
correlation_check = np.corrcoef(y_last_test, np.mean(X_test[:, -1, :], axis=1))[0, 1]
print(f"\n  y_last与输入特征最后时刻均值的相关性: {correlation_check:.4f}")

# 判断3：对比delta的R²和绝对值的R²
if r2_delta < 0.1 and r2_with_last > 0.9:
    print(f"\n⚠️  问题3: Delta R²({r2_delta:.4f}) 远低于绝对值R²({r2_with_last:.4f})")
    print("  这说明高R²主要来自y_last，而非模型的预测能力！")
    leakage_detected = True
else:
    print(f"\n✅ Delta R²({r2_delta:.4f}) 与绝对值R²({r2_with_last:.4f}) 相对合理")

print("\n" + "="*70)
if leakage_detected:
    print("⚠️  结论：存在数据泄露或评估不当的问题！")
    print("\n原因分析：")
    print("1. 高R²主要来自y_last（已知的上一时刻温度），而非模型预测")
    print("2. 模型实际上只需要预测很小的delta值")
    print("3. 即使预测delta=0，也能获得很高的R²")
    print("\n这不是传统意义的数据泄露，但评估方式夸大了模型的预测能力。")
else:
    print("✅ 结论：未发现明显的数据泄露问题")
print("="*70)

# 可视化对比
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 图1：Delta预测对比
ax1 = axes[0, 0]
indices = range(min(500, len(y_delta_test)))
ax1.plot(indices, y_delta_test[:500], label='真实Delta', alpha=0.7)
ax1.plot(indices, predictions_delta_original[:500], label='预测Delta', alpha=0.7)
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Delta=0')
ax1.set_xlabel('样本')
ax1.set_ylabel('温度变化 (°C)')
ax1.set_title('Delta预测对比')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2：Delta分布对比
ax2 = axes[0, 1]
ax2.hist(y_delta_test, bins=50, alpha=0.5, label='真实Delta', density=True)
ax2.hist(predictions_delta_original, bins=50, alpha=0.5, label='预测Delta', density=True)
ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel('温度变化 (°C)')
ax2.set_ylabel('密度')
ax2.set_title('Delta分布对比')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3：绝对值预测散点图
ax3 = axes[1, 0]
ax3.scatter(actuals_absolute[:1000], predictions_abs_with_last[:1000], alpha=0.5, s=10)
min_val = min(actuals_absolute.min(), predictions_abs_with_last.min())
max_val = max(actuals_absolute.max(), predictions_abs_with_last.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
ax3.set_xlabel('真实温度 (°C)')
ax3.set_ylabel('预测温度 (°C)')
ax3.set_title(f'绝对温度预测 (R²={r2_with_last:.4f})')
ax3.grid(True, alpha=0.3)

# 图4：Delta散点图
ax4 = axes[1, 1]
ax4.scatter(y_delta_test[:1000], predictions_delta_original[:1000], alpha=0.5, s=10)
delta_min = min(y_delta_test.min(), predictions_delta_original.min())
delta_max = max(y_delta_test.max(), predictions_delta_original.max())
ax4.plot([delta_min, delta_max], [delta_min, delta_max], 'r--', linewidth=2)
ax4.axhline(y=0, color='green', linestyle='--', alpha=0.3)
ax4.axvline(x=0, color='green', linestyle='--', alpha=0.3)
ax4.set_xlabel('真实Delta (°C)')
ax4.set_ylabel('预测Delta (°C)')
ax4.set_title(f'Delta预测散点图 (R²={r2_delta:.4f})')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
save_path = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\results\data_leakage_analysis.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n分析图表已保存至: {save_path}")
plt.close()
