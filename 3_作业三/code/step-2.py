"""
步骤2: LSTM模型

实验内容：
- 使用长短期记忆网络（LSTM）捕捉时间序列的长期依赖关系
- 模型架构：LSTM(2层,64隐藏单元) -> 全连接层(32) -> 输出层(1)
- 目的：验证LSTM在时序预测任务中相比基线模型的优势
- 输入：过去2小时（12个时间步）× 21个气象特征
- 输出：下一时间点的室外温度（OT）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from utils import set_seed, get_device, load_and_preprocess_data, create_feature_target_sequences, WeatherDataset, plot_predictions, plot_scatter, generate_report

set_seed(42)
device = get_device()
print(f'使用设备: {device}')

# ==================== 数据加载和预处理 ====================

# common utilities imported from utils.py

# ==================== 模型定义 ====================

class LSTMModel(nn.Module):
    """LSTM模型"""
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
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

# ==================== 可视化函数 ====================

# plotting and report generation are provided by utils.py

# ==================== 主函数 ====================

def main():
    print("="*50)
    print("步骤2: LSTM模型")
    print("="*50)
    
    results_dir = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\results'
    docs_dir = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\docs'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    data_path = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\data\weather.csv'
    
    df = load_and_preprocess_data(data_path)
    
    feature_cols = [col for col in df.columns if col not in ['date', 'OT']]
    features = df[feature_cols].values
    target = df['OT'].values.reshape(-1, 1)

    seq_length = 12
    X, y = create_feature_target_sequences(features, target, seq_length)

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
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    train_dataset = WeatherDataset(X_train_scaled, y_train_scaled)
    test_dataset = WeatherDataset(X_test_scaled, y_test_scaled)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    input_size = X_train.shape[2]
    model = LSTMModel(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n开始训练...")
    train_losses = train_model(model, train_loader, criterion, optimizer, epochs=100)
    
    print("\n评估模型...")
    predictions, actuals, metrics = evaluate_model(model, test_loader, scaler_y)
    
    print("\n模型性能:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    plot_predictions(actuals, predictions, "LSTM模型", 
                    os.path.join(results_dir, "step2_predictions.png"))
    plot_scatter(actuals, predictions, "LSTM模型",
                os.path.join(results_dir, "step2_scatter.png"))
    
    generate_report(metrics, "步骤2: LSTM模型",
                   os.path.join(docs_dir, "step2_report.md"))
    
    print("\n" + "="*50)
    print("步骤2完成！")
    print("="*50)

if __name__ == "__main__":
    main()
