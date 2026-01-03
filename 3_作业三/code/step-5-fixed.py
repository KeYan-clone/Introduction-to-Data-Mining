"""
步骤5修正版: 双向LSTM + 周期性时间特征 + 预测变化值（包含OT历史）

实验内容：
- 基于步骤4的双向LSTM和周期性时间特征
- 改进点：预测温度变化值（delta）而非绝对值
- **关键修正**：输入特征包含历史OT值，使模型能学习温度模式
- 预测逻辑：delta = OT(t+1) - OT(t) → OT(t+1) = OT(t) + delta
- 模型架构：BiLSTM(2层,64隐藏单元) -> 全连接层(64) -> 全连接层(32) -> 输出层(1)
- 特征维度：28维（21个气象特征 + OT + 6个时间特征）
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

from utils import set_seed, get_device, load_and_preprocess_data, extract_time_features, create_feature_target_sequences_with_delta, WeatherDataset, plot_predictions, plot_scatter

set_seed(42)
device = get_device()
print(f'使用设备: {device}')

# ==================== 模型定义 ====================

class BiLSTMDeltaModel(nn.Module):
    """双向LSTM模型 - 预测温度变化值"""
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(BiLSTMDeltaModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2因为是双向
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

# ==================== 训练和评估函数 ====================

def train_model(model, train_loader, criterion, optimizer, epochs=100, patience=10):
    """训练模型"""
    model.train()
    train_losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            if len(batch) == 3:
                X_batch, y_delta_batch, _ = batch
            else:
                X_batch, y_delta_batch = batch
            
            X_batch, y_delta_batch = X_batch.to(device), y_delta_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_delta_batch)
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

def evaluate_model(model, test_loader, scaler_delta):
    """评估模型 - 预测delta，然后转换为绝对值"""
    model.eval()
    predictions_delta = []
    actuals_delta = []
    last_temps = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                X_batch, y_delta_batch, y_last_batch = batch
            else:
                X_batch, y_delta_batch = batch
                y_last_batch = None
            
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            predictions_delta.extend(outputs.cpu().numpy())
            actuals_delta.extend(y_delta_batch.numpy())
            
            if y_last_batch is not None:
                last_temps.extend(y_last_batch.numpy())
    
    predictions_delta = np.array(predictions_delta).reshape(-1, 1)
    actuals_delta = np.array(actuals_delta).reshape(-1, 1)
    
    # 将预测的delta从标准化空间转回原始空间
    predictions_delta_original = scaler_delta.inverse_transform(predictions_delta).flatten()
    actuals_delta_original = scaler_delta.inverse_transform(actuals_delta).flatten()
    
    # 通过 last_temp + delta 得到预测的绝对温度
    last_temps = np.array(last_temps)
    predictions_absolute = last_temps + predictions_delta_original
    actuals_absolute = last_temps + actuals_delta_original
    
    return predictions_absolute, actuals_absolute, predictions_delta_original, actuals_delta_original

# ==================== 主函数 ====================

def main():
    print("="*50)
    print("步骤5修正版: 双向LSTM + 时间特征 + 预测变化值（含OT历史）")
    print("="*50)
    
    results_dir = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\results'
    docs_dir = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\docs'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    data_path = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\data\weather.csv'
    
    df = load_and_preprocess_data(data_path)
    
    # 添加时间特征
    df_with_time = extract_time_features(df)

    # 【关键修正】准备数据（包含OT在特征中）
    feature_cols = [col for col in df.columns if col != 'date']  # 包含OT
    time_feature_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    feature_cols_extended = [col for col in feature_cols if col != 'OT'] + time_feature_cols + ['OT']
    features_extended = df_with_time[feature_cols_extended].values
    target = df_with_time['OT'].values.reshape(-1, 1)

    print(f"\n特征维度: {len(feature_cols_extended)}")
    print(f"包含OT历史特征: {'OT' in feature_cols_extended}")

    seq_length = 12
    # 使用新的函数创建序列，返回X, y_delta, y_last
    X, y_delta, y_last = create_feature_target_sequences_with_delta(features_extended, target, seq_length)

    # 划分训练集和测试集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_delta_train, y_delta_test = y_delta[:split_idx], y_delta[split_idx:]
    y_last_train, y_last_test = y_last[:split_idx], y_last[split_idx:]

    # 标准化X特征
    n_features = X_train.shape[2]
    scaler_X = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    scaler_X.fit(X_train_flat)
    X_train_scaled = scaler_X.transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test.shape)

    # 标准化delta值
    scaler_delta = StandardScaler()
    scaler_delta.fit(y_delta_train.reshape(-1, 1))
    y_delta_train_scaled = scaler_delta.transform(y_delta_train.reshape(-1, 1)).flatten()
    y_delta_test_scaled = scaler_delta.transform(y_delta_test.reshape(-1, 1)).flatten()
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"Delta标准化参数 - Mean: {scaler_delta.mean_[0]:.4f}, Std: {scaler_delta.scale_[0]:.4f}")
    
    # 创建数据集和数据加载器
    train_dataset = WeatherDataset(X_train_scaled, y_delta_train_scaled, y_last_train)
    test_dataset = WeatherDataset(X_test_scaled, y_delta_test_scaled, y_last_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 初始化模型
    input_size = X_train.shape[2]
    model = BiLSTMDeltaModel(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n开始训练...")
    train_losses = train_model(model, train_loader, criterion, optimizer, epochs=100)
    
    print("\n评估模型...")
    predictions_absolute, actuals_absolute, predictions_delta, actuals_delta = evaluate_model(model, test_loader, scaler_delta)
    
    # 计算评估指标（基于绝对值）
    mse = mean_squared_error(actuals_absolute, predictions_absolute)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_absolute, predictions_absolute)
    r2 = r2_score(actuals_absolute, predictions_absolute)
    
    # 计算Delta的评估指标
    delta_mse = mean_squared_error(actuals_delta, predictions_delta)
    delta_rmse = np.sqrt(delta_mse)
    delta_mae = mean_absolute_error(actuals_delta, predictions_delta)
    delta_r2 = r2_score(actuals_delta, predictions_delta)
    
    metrics = {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2,
        'Delta_MSE': delta_mse, 'Delta_RMSE': delta_rmse, 
        'Delta_MAE': delta_mae, 'Delta_R2': delta_r2
    }
    
    print("\n【关键】模型性能对比:")
    print("="*50)
    print("绝对温度预测性能:")
    print(f"  MSE: {mse:.4f}, RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}, R²: {r2:.4f}")
    print("\nDelta预测性能（真实预测能力）:")
    print(f"  Delta MSE: {delta_mse:.4f}, Delta RMSE: {delta_rmse:.4f}")
    print(f"  Delta MAE: {delta_mae:.4f}, Delta R²: {delta_r2:.4f}")
    print("="*50)
    
    # Baseline对比
    baseline_predictions = np.zeros_like(actuals_delta)
    baseline_mae = mean_absolute_error(actuals_delta, baseline_predictions)
    baseline_r2 = r2_score(actuals_delta, baseline_predictions)
    
    print(f"\nBaseline（预测delta=0）:")
    print(f"  Baseline Delta MAE: {baseline_mae:.4f}")
    print(f"  Baseline Delta R²: {baseline_r2:.4f}")
    print(f"\n模型相对Baseline改进:")
    print(f"  MAE改进: {(baseline_mae - delta_mae) / baseline_mae * 100:.2f}%")
    if baseline_r2 >= 0:
        print(f"  R²改进: {(delta_r2 - baseline_r2) / max(abs(baseline_r2), 0.01) * 100:.2f}%")
    
    # 绘制预测结果图
    plot_predictions(actuals_absolute, predictions_absolute, "步骤5修正版-绝对温度", 
                    os.path.join(results_dir, "step5_fixed_predictions_absolute.png"))
    plot_scatter(actuals_absolute, predictions_absolute, "步骤5修正版-绝对温度",
                os.path.join(results_dir, "step5_fixed_scatter_absolute.png"))
    
    # 绘制Delta预测结果
    plot_predictions(actuals_delta, predictions_delta, "步骤5修正版-Delta预测", 
                    os.path.join(results_dir, "step5_fixed_predictions_delta.png"), num_points=500)
    plot_scatter(actuals_delta, predictions_delta, "步骤5修正版-Delta预测",
                os.path.join(results_dir, "step5_fixed_scatter_delta.png"))
    
    # 生成报告
    report_content = f"""# 步骤5修正版: 双向LSTM + 时间特征 + 预测变化值（含OT历史）

生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 关键修正

本实验修正了原步骤5的评估问题：
- **修正前**: 输入特征不包含OT历史，导致评估不公平
- **修正后**: 输入特征包含OT历史（28维特征），模型可以学习温度模式
- **评估改进**: 同时报告绝对值和Delta的R²，分离"信息优势"和"模型能力"

## 模型架构

- 输入: 过去12个时间步的28维特征（21个气象特征 + **OT历史** + 6个周期性时间特征）
- BiLSTM: 2层，64隐藏单元，双向结构
- 全连接层: 64 -> 32 -> 1
- Dropout: 0.2
- 激活函数: ReLU

## 性能指标

### 绝对温度预测性能

| 指标 | 值 |
|------|------|
| MSE | {mse:.4f} |
| RMSE | {rmse:.4f} |
| MAE | {mae:.4f} |
| R² | {r2:.4f} |

### Delta预测性能（真实预测能力）

| 指标 | 值 |
|------|------|
| Delta MSE | {delta_mse:.4f} |
| Delta RMSE | {delta_rmse:.4f} |
| Delta MAE | {delta_mae:.4f} |
| **Delta R²** | **{delta_r2:.4f}** |

### Baseline对比

| 指标 | Baseline (delta=0) | 模型 | 改进 |
|------|-------------------|------|------|
| Delta MAE | {baseline_mae:.4f} | {delta_mae:.4f} | {(baseline_mae - delta_mae) / baseline_mae * 100:.2f}% |
| Delta R² | {baseline_r2:.4f} | {delta_r2:.4f} | - |

## 与其他步骤对比

| 步骤 | 预测目标 | 包含OT历史 | R²（可比） | MAE |
|------|---------|-----------|-----------|-----|
| 步骤4 | 绝对值 | ✅ | 0.5688 | 8.70°C |
| 步骤5原版 | Delta | ❌ | ~~0.9734~~（虚高） | 2.11°C |
| **步骤5修正版** | **Delta** | **✅** | **{delta_r2:.4f}** | **{delta_mae:.4f}°C** |

## 实验结论

1. **修正后的真实性能**：
   - Delta R² = {delta_r2:.4f}（真实预测能力）
   - 相比步骤4的R²=0.57，{'提升' if delta_r2 > 0.57 else '下降'}了{abs(delta_r2 - 0.57) / 0.57 * 100:.1f}%

2. **预测Delta vs 预测绝对值**：
   - Delta MAE更小是因为预测目标本身更小
   - 真实预测能力应看Delta R²，而非绝对值R²

3. **包含OT历史的重要性**：
   - 模型可以学习温度的时间模式
   - 评估更公平，不依赖外部的y_last信息

## 图表

- 绝对温度预测对比图: `results/step5_fixed_predictions_absolute.png`
- 绝对温度散点图: `results/step5_fixed_scatter_absolute.png`
- Delta预测对比图: `results/step5_fixed_predictions_delta.png`
- Delta散点图: `results/step5_fixed_scatter_delta.png`

## 下一步建议

1. **如果Delta R² > 0.57**: 说明预测delta确实比预测绝对值更好
2. **如果Delta R² < 0.57**: 考虑继续使用步骤4的绝对值预测
3. **进一步改进**: 添加注意力机制、尝试Transformer等
"""
    
    report_path = os.path.join(docs_dir, "step5_fixed_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("\n" + "="*50)
    print("步骤5修正版完成！")
    print(f"结果已保存至: {results_dir}")
    print(f"报告已保存至: {report_path}")
    print("="*50)

if __name__ == "__main__":
    main()
