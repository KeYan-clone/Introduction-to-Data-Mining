"""
步骤7-无泄露版本: 双向LSTM + 周期性时间特征（完全不使用OT统计特征）

实验目的：
- 验证step-7中OT统计特征是否导致数据泄露
- 保持step-4的所有设置，只验证纯净的双向LSTM效果
- 如果性能接近step-4（R²≈0.57），说明step-7的高性能来自OT特征泄露
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
                   extract_time_features, create_feature_target_sequences, 
                   WeatherDataset, plot_predictions, plot_scatter)

set_seed(42)
device = get_device()
print(f'使用设备: {device}')

# ==================== 模型定义 ====================

class BiLSTMModel(nn.Module):
    """双向LSTM模型"""
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
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

def train_model(model, train_loader, criterion, optimizer, epochs=100, patience=10):
    """训练模型"""
    model.train()
    train_losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return train_losses

def evaluate_model(model, test_loader, scaler_y):
    """评估模型"""
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
    
    return predictions, actuals, {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

def main():
    print("="*70)
    print("步骤7-无泄露验证: 双向LSTM + 时间特征（不使用OT统计特征）")
    print("="*70)
    
    data_path = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\data\weather.csv'
    
    df = load_and_preprocess_data(data_path)
    df_with_time = extract_time_features(df)

    # 只使用原始特征和时间特征，完全不使用OT的统计特征
    feature_cols = [col for col in df.columns if col not in ['date', 'OT']]
    time_feature_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    feature_cols_extended = feature_cols + time_feature_cols
    
    features_extended = df_with_time[feature_cols_extended].values
    target = df_with_time['OT'].values.reshape(-1, 1)
    
    print(f"\n使用特征:")
    print(f"  - 原始气象特征: {len(feature_cols)}个")
    print(f"  - 周期性时间特征: {len(time_feature_cols)}个")
    print(f"  - 总特征维度: {len(feature_cols_extended)}个")
    print(f"  ⚠️ 完全不使用OT的统计特征（移动平均、标准差、差分等）")

    seq_length = 12
    X, y = create_feature_target_sequences(features_extended, target, seq_length)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    n_features = X_train.shape[2]
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    scaler_X.fit(X_train_flat)
    X_train_scaled = scaler_X.transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test.shape)

    scaler_y.fit(y_train.reshape(-1,1))
    y_train_scaled = scaler_y.transform(y_train.reshape(-1,1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1,1)).flatten()
    
    print(f"\n训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    train_dataset = WeatherDataset(X_train_scaled, y_train_scaled)
    test_dataset = WeatherDataset(X_test_scaled, y_test_scaled)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    input_size = X_train.shape[2]
    model = BiLSTMModel(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n开始训练...")
    train_losses = train_model(model, train_loader, criterion, optimizer, epochs=100)
    
    print("\n评估模型...")
    predictions, actuals, metrics = evaluate_model(model, test_loader, scaler_y)
    
    print("\n" + "="*70)
    print("【无泄露版本】模型性能:")
    print("="*70)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 对比分析
    step4_metrics = {'MAE': 8.70, 'RMSE': 14.84, 'R2': 0.5688}
    step7_with_ot_features = {'MAE': 5.51, 'RMSE': 9.11, 'R2': 0.8377}
    
    print("\n" + "="*70)
    print("对比分析：数据泄露验证")
    print("="*70)
    print(f"\nStep 4 (双向LSTM+时间特征):")
    print(f"  MAE: {step4_metrics['MAE']:.2f}°C, R²: {step4_metrics['R2']:.4f}")
    
    print(f"\nStep 7 无泄露版本 (相同配置，不用OT统计特征):")
    print(f"  MAE: {metrics['MAE']:.2f}°C, R²: {metrics['R2']:.4f}")
    
    print(f"\nStep 7 原版本 (使用OT的移动平均和标准差):")
    print(f"  MAE: {step7_with_ot_features['MAE']:.2f}°C, R²: {step7_with_ot_features['R2']:.4f}")
    
    print("\n" + "="*70)
    print("结论:")
    print("="*70)
    
    r2_diff_no_leak = abs(metrics['R2'] - step4_metrics['R2'])
    r2_diff_with_ot = abs(step7_with_ot_features['R2'] - step4_metrics['R2'])
    
    if r2_diff_no_leak < 0.05:
        print("✅ 无泄露版本的性能与Step 4接近（R²差异<5%）")
        print("   说明：双向LSTM本身相比单向LSTM没有显著提升")
    else:
        print(f"⚠️ 无泄露版本的R²与Step 4相差{r2_diff_no_leak:.2%}")
    
    if r2_diff_with_ot > 0.15:
        print(f"\n❌ 使用OT统计特征后R²提升了{r2_diff_with_ot:.2%}")
        print("   **结论：step-7的高性能主要来自OT统计特征的信息泄露！**")
        print("   原因：")
        print("   1. 移动平均（temp_ma_*）包含OT的历史值")
        print("   2. 温度具有高度连续性，历史均值≈当前值")
        print("   3. 模型实际上在\"记忆\"而非\"预测\"")
    else:
        print(f"\n✅ 使用OT统计特征提升了{r2_diff_with_ot:.2%}，属于合理范围")
    
    print("\n" + "="*70)
    print("推荐:")
    print("="*70)
    print("对于时序预测任务，不应使用目标变量的历史统计量作为特征！")
    print("这会导致模型过度依赖历史值，而非学习真正的预测规律。")
    print("="*70)

if __name__ == "__main__":
    main()
