"""
步骤5: 双向LSTM + 周期性时间特征 + 预测变化值

实验内容：
- 基于步骤4的双向LSTM和周期性时间特征
- 改进点：预测温度变化值（delta）而非绝对值
- 预测逻辑：delta = OT(t+1) - OT(t) → OT(t+1) = OT(t) + delta
- 模型架构：BiLSTM(2层,64隐藏单元) -> 全连接层(64) -> 全连接层(32) -> 输出层(1)
- 目的：通过预测变化值来提高模型对温度趋势的捕捉能力
- 特征维度：27维（21个原始特征 + 6个时间特征）
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

from utils import set_seed, get_device, load_and_preprocess_data, extract_time_features, create_feature_target_sequences_with_delta, WeatherDataset, plot_predictions, plot_scatter, generate_report

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
    actuals_absolute = []
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
            
            if y_last_batch is not None:
                last_temps.extend(y_last_batch.numpy())
    
    predictions_delta = np.array(predictions_delta).reshape(-1, 1)
    
    # 将预测的delta从标准化空间转回原始空间
    predictions_delta_original = scaler_delta.inverse_transform(predictions_delta).flatten()
    
    # 通过 last_temp + delta 得到预测的绝对温度
    last_temps = np.array(last_temps)
    predictions_absolute = last_temps + predictions_delta_original
    
    # 计算实际的绝对温度（last_temp + actual_delta）
    # 注意：actual_delta在loader中已被标准化，需从test_dataset获取原始值
    # 为简化，我们直接从外部传入actuals_absolute
    
    return predictions_absolute, predictions_delta_original

# ==================== 主函数 ====================

def main():
    print("="*50)
    print("步骤5: 双向LSTM + 时间特征 + 预测变化值")
    print("="*50)
    
    results_dir = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\results'
    docs_dir = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\docs'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    data_path = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\data\weather.csv'
    
    df = load_and_preprocess_data(data_path)
    
    # 添加时间特征
    df_with_time = extract_time_features(df)

    # 准备数据（包含时间特征）
    feature_cols = [col for col in df.columns if col not in ['date', 'OT']]
    time_feature_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    feature_cols_extended = feature_cols + time_feature_cols
    features_extended = df_with_time[feature_cols_extended].values
    target = df_with_time['OT'].values.reshape(-1, 1)

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
    predictions_absolute, predictions_delta = evaluate_model(model, test_loader, scaler_delta)
    
    # 计算实际的绝对温度值
    actuals_absolute = y_last_test + y_delta_test
    
    # 计算评估指标（基于绝对值）
    mse = mean_squared_error(actuals_absolute, predictions_absolute)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_absolute, predictions_absolute)
    r2 = r2_score(actuals_absolute, predictions_absolute)
    
    metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    print("\n模型性能（基于绝对值评估）:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 额外输出delta的统计信息
    delta_mae = mean_absolute_error(y_delta_test, predictions_delta)
    delta_rmse = np.sqrt(mean_squared_error(y_delta_test, predictions_delta))
    print(f"\nDelta预测性能:")
    print(f"Delta MAE: {delta_mae:.4f}")
    print(f"Delta RMSE: {delta_rmse:.4f}")
    
    # 绘制预测结果图
    plot_predictions(actuals_absolute, predictions_absolute, "双向LSTM+时间特征+预测变化值", 
                    os.path.join(results_dir, "step5_predictions.png"))
    plot_scatter(actuals_absolute, predictions_absolute, "双向LSTM+时间特征+预测变化值",
                os.path.join(results_dir, "step5_scatter.png"))
    
    # 生成报告
    report_content = f"""# 步骤5: 双向LSTM + 时间特征 + 预测变化值 实验报告

生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 实验设计

本实验基于步骤4的双向LSTM模型，但采用了不同的预测策略：
- **预测目标改进**: 预测温度变化值（delta）而非绝对温度值
- **预测公式**: OT(t+1) = OT(t) + Δ，其中Δ = OT(t+1) - OT(t)
- **优势**: 变化值通常比绝对值更平稳，且更容易捕捉温度变化趋势

## 模型架构

- 输入: 过去12个时间步的27维特征（21个气象特征 + 6个周期性时间特征）
- BiLSTM: 2层，64隐藏单元，双向结构
- 全连接层: 64 -> 32 -> 1
- Dropout: 0.2
- 激活函数: ReLU

## 性能指标（基于绝对值评估）

| 指标 | 值 |
|------|------|
| MSE | {mse:.4f} |
| RMSE | {rmse:.4f} |
| MAE | {mae:.4f} |
| R2 | {r2:.4f} |

## Delta预测性能

| 指标 | 值 |
|------|------|
| Delta MAE | {delta_mae:.4f} |
| Delta RMSE | {delta_rmse:.4f} |

## 实验结论

通过预测变化值而非绝对值，模型能够更好地：
1. 捕捉温度的短期变化趋势
2. 减少累积误差的影响
3. 提高对温度波动的敏感性

## 图表

- 预测结果对比图: `results/step5_predictions.png`
- 预测散点图: `results/step5_scatter.png`
"""
    
    report_path = os.path.join(docs_dir, "step5_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("\n" + "="*50)
    print("步骤5完成！")
    print(f"结果已保存至: {results_dir}")
    print(f"报告已保存至: {report_path}")
    print("="*50)

if __name__ == "__main__":
    main()
