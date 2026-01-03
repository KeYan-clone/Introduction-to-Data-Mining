"""
步骤1: 基线模型 - 全连接神经网络

实验内容：
- 使用简单的全连接神经网络作为基线模型
- 模型架构：输入层 -> 隐藏层(64) -> 隐藏层(32) -> 输出层(1)
- 目的：建立性能基准，为后续实验提供对比基础
- 输入：过去2小时（12个时间点）× 21个气象特征
- 输出：下一时间点的室外温度（OT）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from utils import set_seed, get_device, load_and_preprocess_data, create_feature_target_sequences, WeatherDataset, plot_predictions, plot_scatter, generate_report

set_seed(42)
device = get_device()
print(f'使用设备: {device}')

# ==================== 数据加载和预处理 ====================

# common utilities imported from utils.py

# ==================== 模型定义 ====================

class BaselineModel(nn.Module):
    """基线模型：简单的全连接网络"""
    def __init__(self, input_size, hidden_size=64):
        super(BaselineModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(input_size * 12, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

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
    print("步骤1: 基线模型 - 全连接神经网络")
    print("="*50)
    
    # 创建目录
    results_dir = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\results'
    docs_dir = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\docs'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    # 数据路径
    data_path = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\data\weather.csv'
    
    # 加载数据
    df = load_and_preprocess_data(data_path)
    
    # 准备数据（不包含时间特征）
    feature_cols = [col for col in df.columns if col not in ['date', 'OT']]
    features = df[feature_cols].values
    target = df['OT'].values.reshape(-1, 1)

    # 先按时间创建序列（不做全量缩放，防止泄露），随后按时间划分训练/测试，再在训练集上拟合 scaler
    seq_length = 12
    X, y = create_feature_target_sequences(features, target, seq_length)

    # 划分训练集和测试集 (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 在训练集上拟合 scaler 并转换训练/测试集
    n_features = X_train.shape[2]
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    scaler_X.fit(X_train_flat)
    X_train_scaled = scaler_X.transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test.shape)

    scaler_y.fit(y_train.reshape(-1, 1))
    y_train_scaled = scaler_y.transform(y_train.reshape(-1,1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1,1)).flatten()
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 创建数据加载器
    train_dataset = WeatherDataset(X_train_scaled, y_train_scaled)
    test_dataset = WeatherDataset(X_test_scaled, y_test_scaled)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 训练模型
    input_size = X_train.shape[2]
    model = BaselineModel(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n开始训练...")
    train_losses = train_model(model, train_loader, criterion, optimizer, epochs=100)
    
    # 评估
    print("\n评估模型...")
    predictions, actuals, metrics = evaluate_model(model, test_loader, scaler_y)
    
    print("\n模型性能:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 可视化
    plot_predictions(actuals, predictions, "基线模型", 
                    os.path.join(results_dir, "step1_predictions.png"))
    plot_scatter(actuals, predictions, "基线模型",
                os.path.join(results_dir, "step1_scatter.png"))
    
    # 生成报告
    generate_report(metrics, "步骤1: 基线模型",
                   os.path.join(docs_dir, "step1_report.md"))
    
    print("\n" + "="*50)
    print("步骤1完成！")
    print("="*50)

if __name__ == "__main__":
    main()
