"""
步骤7数据泄露深度分析 - 使用完整序列模型测试

检查修正后真实的LSTM性能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from utils import set_seed, get_device, load_and_preprocess_data, extract_time_features, create_feature_target_sequences, WeatherDataset

set_seed(42)
device = get_device()

print("="*70)
print("步骤7深度数据泄露分析 - 完整LSTM模型测试")
print("="*70)

# ==================== 定义模型 ====================
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

def train_and_evaluate(X_train, y_train, X_test, y_test, scaler_y, model_name, epochs=30):
    """训练并评估模型"""
    train_dataset = WeatherDataset(X_train, y_train)
    test_dataset = WeatherDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    input_size = X_train.shape[2]
    model = BiLSTMModel(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    model.train()
    print(f"  训练中...", end='', flush=True)
    for epoch in range(epochs):
        if (epoch + 1) % 10 == 0:
            print(f" {epoch+1}/{epochs}", end='', flush=True)
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    print(" 完成")
    
    # 评估
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())
    
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    predictions = scaler_y.inverse_transform(predictions).flatten()
    actuals = scaler_y.inverse_transform(actuals).flatten()
    
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    print(f"\n{model_name}:")
    print(f"  MAE: {mae:.2f}°C")
    print(f"  RMSE: {rmse:.2f}°C")
    print(f"  R²: {r2:.4f}")
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'predictions': predictions, 'actuals': actuals}

# ==================== 加载和准备数据 ====================
print("\n1. 加载数据...")
data_path = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\data\weather.csv'
df = load_and_preprocess_data(data_path)
df_with_time = extract_time_features(df)

print(f"数据集大小: {len(df)} 条记录")

# ==================== 测试1: 错误方式（当前步骤7） ====================
print("\n" + "="*70)
print("测试1: 错误方式 - rolling() 包含当前值")
print("="*70)

df_wrong = df_with_time.copy()
df_wrong['temp_ma_6'] = df_wrong['OT'].rolling(window=6, min_periods=1).mean()
df_wrong['temp_ma_12'] = df_wrong['OT'].rolling(window=12, min_periods=1).mean()
df_wrong['temp_ma_36'] = df_wrong['OT'].rolling(window=36, min_periods=1).mean()
df_wrong['temp_std_6'] = df_wrong['OT'].rolling(window=6, min_periods=1).std()
df_wrong['temp_std_12'] = df_wrong['OT'].rolling(window=12, min_periods=1).std()
df_wrong['temp_diff_1'] = df_wrong['OT'].diff(1).fillna(0)
df_wrong['temp_diff_6'] = df_wrong['OT'].diff(6).fillna(0)
df_wrong['temp_diff2'] = df_wrong['temp_diff_1'].diff(1).fillna(0)
df_wrong = df_wrong.fillna(method='bfill').fillna(method='ffill')

feature_cols = [col for col in df.columns if col not in ['date', 'OT']]
time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
stat_features = ['temp_ma_6', 'temp_ma_12', 'temp_ma_36', 
                'temp_std_6', 'temp_std_12',
                'temp_diff_1', 'temp_diff_6', 'temp_diff2']
all_features = feature_cols + time_features + stat_features

features_wrong = df_wrong[all_features].values
target = df_wrong['OT'].values.reshape(-1, 1)

seq_length = 12
X_wrong, y_wrong = create_feature_target_sequences(features_wrong, target, seq_length)

split_idx = int(len(X_wrong) * 0.8)
X_train_wrong, X_test_wrong = X_wrong[:split_idx], X_wrong[split_idx:]
y_train_wrong, y_test_wrong = y_wrong[:split_idx], y_wrong[split_idx:]

# 标准化
scaler_X_wrong = StandardScaler()
scaler_y_wrong = StandardScaler()

X_train_wrong_flat = X_train_wrong.reshape(-1, X_train_wrong.shape[2])
X_test_wrong_flat = X_test_wrong.reshape(-1, X_test_wrong.shape[2])

scaler_X_wrong.fit(X_train_wrong_flat)
X_train_wrong_scaled = scaler_X_wrong.transform(X_train_wrong_flat).reshape(X_train_wrong.shape)
X_test_wrong_scaled = scaler_X_wrong.transform(X_test_wrong_flat).reshape(X_test_wrong.shape)

scaler_y_wrong.fit(y_train_wrong.reshape(-1,1))
y_train_wrong_scaled = scaler_y_wrong.transform(y_train_wrong.reshape(-1,1)).flatten()
y_test_wrong_scaled = scaler_y_wrong.transform(y_test_wrong.reshape(-1,1)).flatten()

results_wrong = train_and_evaluate(X_train_wrong_scaled, y_train_wrong_scaled, 
                                   X_test_wrong_scaled, y_test_wrong_scaled, 
                                   scaler_y_wrong, "错误方式（含泄露）")

# ==================== 测试2: 正确方式（使用shift） ====================
print("\n" + "="*70)
print("测试2: 正确方式 - 使用shift(1)确保只用历史数据")
print("="*70)

df_correct = df_with_time.copy()
# 使用shift(1)确保统计特征只包含历史数据
df_correct['temp_ma_6'] = df_correct['OT'].rolling(window=6, min_periods=1).mean().shift(1)
df_correct['temp_ma_12'] = df_correct['OT'].rolling(window=12, min_periods=1).mean().shift(1)
df_correct['temp_ma_36'] = df_correct['OT'].rolling(window=36, min_periods=1).mean().shift(1)
df_correct['temp_std_6'] = df_correct['OT'].rolling(window=6, min_periods=1).std().shift(1)
df_correct['temp_std_12'] = df_correct['OT'].rolling(window=12, min_periods=1).std().shift(1)
df_correct['temp_diff_1'] = df_correct['OT'].diff(1)  # diff已经是滞后的
df_correct['temp_diff_6'] = df_correct['OT'].diff(6)
df_correct['temp_diff2'] = df_correct['temp_diff_1'].diff(1)
df_correct = df_correct.fillna(method='bfill').fillna(method='ffill')

features_correct = df_correct[all_features].values
X_correct, y_correct = create_feature_target_sequences(features_correct, target, seq_length)

X_train_correct, X_test_correct = X_correct[:split_idx], X_correct[split_idx:]
y_train_correct, y_test_correct = y_correct[:split_idx], y_correct[split_idx:]

# 标准化
scaler_X_correct = StandardScaler()
scaler_y_correct = StandardScaler()

X_train_correct_flat = X_train_correct.reshape(-1, X_train_correct.shape[2])
X_test_correct_flat = X_test_correct.reshape(-1, X_test_correct.shape[2])

scaler_X_correct.fit(X_train_correct_flat)
X_train_correct_scaled = scaler_X_correct.transform(X_train_correct_flat).reshape(X_train_correct.shape)
X_test_correct_scaled = scaler_X_correct.transform(X_test_correct_flat).reshape(X_test_correct.shape)

scaler_y_correct.fit(y_train_correct.reshape(-1,1))
y_train_correct_scaled = scaler_y_correct.transform(y_train_correct.reshape(-1,1)).flatten()
y_test_correct_scaled = scaler_y_correct.transform(y_test_correct.reshape(-1,1)).flatten()

results_correct = train_and_evaluate(X_train_correct_scaled, y_train_correct_scaled,
                                    X_test_correct_scaled, y_test_correct_scaled,
                                    scaler_y_correct, "正确方式（无泄露）")

# ==================== 测试3: 步骤4基线对比 ====================
print("\n" + "="*70)
print("测试3: 步骤4基线（无统计特征）")
print("="*70)

features_step4 = df_with_time[feature_cols + time_features].values
X_step4, y_step4 = create_feature_target_sequences(features_step4, target, seq_length)

X_train_step4, X_test_step4 = X_step4[:split_idx], X_step4[split_idx:]
y_train_step4, y_test_step4 = y_step4[:split_idx], y_step4[split_idx:]

scaler_X_step4 = StandardScaler()
scaler_y_step4 = StandardScaler()

X_train_step4_flat = X_train_step4.reshape(-1, X_train_step4.shape[2])
X_test_step4_flat = X_test_step4.reshape(-1, X_test_step4.shape[2])

scaler_X_step4.fit(X_train_step4_flat)
X_train_step4_scaled = scaler_X_step4.transform(X_train_step4_flat).reshape(X_train_step4.shape)
X_test_step4_scaled = scaler_X_step4.transform(X_test_step4_flat).reshape(X_test_step4.shape)

scaler_y_step4.fit(y_train_step4.reshape(-1,1))
y_train_step4_scaled = scaler_y_step4.transform(y_train_step4.reshape(-1,1)).flatten()
y_test_step4_scaled = scaler_y_step4.transform(y_test_step4.reshape(-1,1)).flatten()

results_step4 = train_and_evaluate(X_train_step4_scaled, y_train_step4_scaled,
                                  X_test_step4_scaled, y_test_step4_scaled,
                                  scaler_y_step4, "步骤4（BiLSTM+时间）")

# ==================== 生成对比图 ====================
print("\n" + "="*70)
print("生成对比可视化")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: MAE对比
ax1 = axes[0, 0]
models = ['错误方式\n(含泄露)', '正确方式\n(shift)', '步骤4\n(基线)']
mae_values = [results_wrong['MAE'], results_correct['MAE'], results_step4['MAE']]
colors = ['red', 'orange', 'green']
bars = ax1.bar(models, mae_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
for bar, val in zip(bars, mae_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}°C',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.set_ylabel('MAE (°C)', fontsize=11, fontweight='bold')
ax1.set_title('平均绝对误差对比', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 图2: R²对比
ax2 = axes[0, 1]
r2_values = [results_wrong['R2'], results_correct['R2'], results_step4['R2']]
bars = ax2.bar(models, r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
for bar, val in zip(bars, r2_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.4f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
ax2.set_ylabel('R² Score', fontsize=11, fontweight='bold')
ax2.set_title('R²决定系数对比', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.0)
ax2.grid(axis='y', alpha=0.3)

# 图3: 预测对比（错误方式）
ax3 = axes[1, 0]
sample_size = 200
ax3.plot(results_wrong['actuals'][:sample_size], label='实际', linewidth=2, alpha=0.7)
ax3.plot(results_wrong['predictions'][:sample_size], label='预测', linewidth=2, alpha=0.7)
ax3.set_xlabel('样本索引', fontsize=10)
ax3.set_ylabel('温度 (°C)', fontsize=10)
ax3.set_title(f'错误方式预测 (R²={results_wrong["R2"]:.4f})\n⚠️ 存在数据泄露', 
             fontsize=11, fontweight='bold', color='red')
ax3.legend()
ax3.grid(alpha=0.3)

# 图4: 预测对比（正确方式）
ax4 = axes[1, 1]
ax4.plot(results_correct['actuals'][:sample_size], label='实际', linewidth=2, alpha=0.7)
ax4.plot(results_correct['predictions'][:sample_size], label='预测', linewidth=2, alpha=0.7)
ax4.set_xlabel('样本索引', fontsize=10)
ax4.set_ylabel('温度 (°C)', fontsize=10)
ax4.set_title(f'正确方式预测 (R²={results_correct["R2"]:.4f})\n✅ 无数据泄露', 
             fontsize=11, fontweight='bold', color='green')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
results_dir = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\results'
plt.savefig(f'{results_dir}/step7_leakage_lstm_analysis.png', dpi=300, bbox_inches='tight')
print("✅ 已生成: step7_leakage_lstm_analysis.png")
plt.close()

# ==================== 最终结论 ====================
print("\n" + "="*70)
print("🔍 完整LSTM模型数据泄露分析结论")
print("="*70)

print("\n📊 性能对比:")
print(f"\n{'模型':<20} {'MAE (°C)':<12} {'RMSE (°C)':<12} {'R²':<10}")
print("-" * 54)
print(f"{'错误方式(含泄露)':<20} {results_wrong['MAE']:<12.2f} {results_wrong['RMSE']:<12.2f} {results_wrong['R2']:<10.4f}")
print(f"{'正确方式(shift)':<20} {results_correct['MAE']:<12.2f} {results_correct['RMSE']:<12.2f} {results_correct['R2']:<10.4f}")
print(f"{'步骤4(基线)':<20} {results_step4['MAE']:<12.2f} {results_step4['RMSE']:<12.2f} {results_step4['R2']:<10.4f}")

improvement_wrong = (results_step4['MAE'] - results_wrong['MAE']) / results_step4['MAE'] * 100
improvement_correct = (results_step4['MAE'] - results_correct['MAE']) / results_step4['MAE'] * 100

print(f"\n📈 相对步骤4的改进:")
print(f"  错误方式: MAE改进 {improvement_wrong:+.1f}%")
print(f"  正确方式: MAE改进 {improvement_correct:+.1f}%")

print(f"\n✅ 最终结论:")
if results_correct['R2'] > results_step4['R2'] * 1.05:
    print(f"  ✅ 修正后的特征工程仍然有效！")
    print(f"  ✅ R²从{results_step4['R2']:.4f}提升到{results_correct['R2']:.4f}")
    print(f"  ✅ MAE从{results_step4['MAE']:.2f}°C降低到{results_correct['MAE']:.2f}°C")
    print(f"  ✅ 推荐使用修正后的步骤7")
else:
    print(f"  ⚠️ 修正后性能与步骤4相当")
    print(f"  ⚠️ 统计特征未带来显著提升")
    print(f"  ⚠️ 步骤7的高性能主要来自数据泄露")
    print(f"  ⚠️ 建议继续使用步骤4作为最佳模型")

print("\n" + "="*70)
