"""
时间序列预测任务：基于过去2小时的天气数据预测室外温度（OT）

数据集：weather.csv (德国某气象站半年内的21个气象学指标)
- 数据点总数：26200
- 特征维度：21
- 采样间隔：10分钟
- 任务：使用过去2小时（12个时间点）预测下一时间点的室外温度（OT）
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')


class WeatherDataset(Dataset):
    """时间序列数据集"""
    
    def __init__(self, data, window_size=12):
        self.data = data
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size, -1]
        return torch.FloatTensor(x), torch.FloatTensor([y])


class LSTMPredictor(nn.Module):
    """基于LSTM的时间序列预测模型"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class GRUPredictor(nn.Module):
    """基于GRU的时间序列预测模型"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(GRUPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        gru_out, h_n = self.gru(x)
        last_output = gru_out[:, -1, :]
        
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def load_and_preprocess_data(file_path, window_size=12, train_ratio=0.8):
    """加载并预处理数据"""
    print("=" * 50)
    print("1. 数据加载与预处理")
    print("=" * 50)
    
    # 读取数据
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except:
            df = pd.read_csv(file_path, encoding='latin-1')
    print(f"数据形状: {df.shape}")
    print(f"特征列: {df.columns.tolist()}")
    
    # 检查缺失值
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n缺失值统计:\n{missing[missing > 0]}")
        df.ffill(inplace=True)
        df.bfill(inplace=True)
    else:
        print("无缺失值")
    
    # 移除日期列
    features = df.drop(columns=['date']).values
    print(f"特征数据形状: {features.shape}")
    
    # 数据标准化
    print("\n进行特征标准化...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 划分数据集
    total_len = len(features_scaled)
    train_len = int(total_len * train_ratio * 0.9)
    val_len = int(total_len * train_ratio * 0.1)
    test_len = total_len - train_len - val_len
    
    print(f"\n数据集划分:")
    print(f"训练集: {train_len} 样本")
    print(f"验证集: {val_len} 样本")
    print(f"测试集: {test_len} 样本")
    
    train_data = features_scaled[:train_len]
    val_data = features_scaled[train_len:train_len + val_len]
    test_data = features_scaled[train_len + val_len:]
    
    train_dataset = WeatherDataset(train_data, window_size)
    val_dataset = WeatherDataset(val_data, window_size)
    test_dataset = WeatherDataset(test_data, window_size)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, scaler


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, model_save_path='models/best_model.pth'):
    """训练模型"""
    print("\n" + "=" * 50)
    print("2. 模型训练")
    print("=" * 50)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch+1}')
                break
    
    model.load_state_dict(torch.load(model_save_path))
    print(f'\n最佳验证损失: {best_val_loss:.6f}')
    
    return model, train_losses, val_losses


def evaluate_model(model, test_loader, scaler):
    """评估模型"""
    print("\n" + "=" * 50)
    print("3. 模型评估")
    print("=" * 50)
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    predictions = np.array(predictions).reshape(-1)
    actuals = np.array(actuals).reshape(-1)
    
    # 反标准化
    ot_mean = scaler.mean_[-1]
    ot_std = scaler.scale_[-1]
    
    predictions_original = predictions * ot_std + ot_mean
    actuals_original = actuals * ot_std + ot_mean
    
    # 计算评估指标
    mse = mean_squared_error(actuals_original, predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_original, predictions_original)
    r2 = r2_score(actuals_original, predictions_original)
    mape = np.mean(np.abs((actuals_original - predictions_original) / actuals_original)) * 100
    
    print(f'\n测试集评估指标:')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R²: {r2:.4f}')
    print(f'MAPE: {mape:.2f}%')
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
    
    return predictions_original, actuals_original, metrics


def plot_individual_results(train_losses, val_losses, predictions, actuals, output_dir):
    """
    生成单独的图表文件
    
    Args:
        train_losses: 训练损失
        val_losses: 验证损失
        predictions: 预测值
        actuals: 真实值
        output_dir: 输出目录
    """
    print("\n" + "=" * 50)
    print("4. 生成可视化图表")
    print("=" * 50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 图1: 训练和验证损失
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.8, linewidth=2)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, '01_training_loss.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.close()
    
    # 图2: 预测值 vs 真实值（时间序列，前500个点）
    sample_size = min(500, len(predictions))
    plt.figure(figsize=(14, 6))
    plt.plot(actuals[:sample_size], label='Actual', alpha=0.7, linewidth=1.5)
    plt.plot(predictions[:sample_size], label='Predicted', alpha=0.7, linewidth=1.5)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Temperature (OT)', fontsize=12)
    plt.title(f'Prediction vs Actual (First {sample_size} Points)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, '02_prediction_vs_actual.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.close()
    
    # 图3: 散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.3, s=10)
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Temperature', fontsize=12)
    plt.ylabel('Predicted Temperature', fontsize=12)
    plt.title('Predicted vs Actual Scatter Plot', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    save_path = os.path.join(output_dir, '03_scatter_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.close()
    
    # 图4: 误差分布
    errors = predictions - actuals
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    plt.xlabel('Prediction Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Error Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_path = os.path.join(output_dir, '04_error_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.close()
    
    # 图5: 误差随时间变化（前500个点）
    plt.figure(figsize=(14, 6))
    plt.plot(errors[:sample_size], alpha=0.7, linewidth=1, color='orange')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Prediction Error', fontsize=12)
    plt.title(f'Prediction Error Over Time (First {sample_size} Points)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, '05_error_over_time.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.close()
    
    print(f"\n所有图表已保存到: {output_dir}")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("时间序列预测任务：基于过去2小时天气数据预测室外温度（OT）")
    print("=" * 70)
    
    # 超参数
    WINDOW_SIZE = 12
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # 路径配置
    data_path = '../data/weather.csv'
    
    if not os.path.exists(data_path):
        print(f"错误：数据文件 '{data_path}' 不存在！")
        return
    
    # 加载数据
    train_loader, val_loader, test_loader, scaler = load_and_preprocess_data(
        data_path, window_size=WINDOW_SIZE
    )
    
    sample_x, _ = next(iter(train_loader))
    input_size = sample_x.shape[2]
    print(f"\n输入特征维度: {input_size}")
    print(f"时间窗口大小: {WINDOW_SIZE}")
    
    # 训练LSTM模型
    print("\n" + "=" * 70)
    print("训练 LSTM 模型")
    print("=" * 70)
    
    lstm_model = LSTMPredictor(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    print(f"\n模型结构:\n{lstm_model}")
    print(f"\n模型参数总数: {sum(p.numel() for p in lstm_model.parameters()):,}")
    
    lstm_model, lstm_train_losses, lstm_val_losses = train_model(
        lstm_model, train_loader, val_loader, 
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
        model_save_path='../models/basic_lstm.pth'
    )
    
    lstm_predictions, lstm_actuals, lstm_metrics = evaluate_model(lstm_model, test_loader, scaler)
    plot_individual_results(lstm_train_losses, lstm_val_losses, lstm_predictions, lstm_actuals, '../results/basic_lstm')
    
    # 训练GRU模型
    print("\n" + "=" * 70)
    print("训练 GRU 模型")
    print("=" * 70)
    
    gru_model = GRUPredictor(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    print(f"\n模型参数总数: {sum(p.numel() for p in gru_model.parameters()):,}")
    
    gru_model, gru_train_losses, gru_val_losses = train_model(
        gru_model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
        model_save_path='../models/basic_gru.pth'
    )
    
    gru_predictions, gru_actuals, gru_metrics = evaluate_model(gru_model, test_loader, scaler)
    plot_individual_results(gru_train_losses, gru_val_losses, gru_predictions, gru_actuals, '../results/basic_gru')
    
    # 模型对比
    print("\n" + "=" * 70)
    print("模型性能对比")
    print("=" * 70)
    
    comparison_df = pd.DataFrame({
        'Model': ['LSTM', 'GRU'],
        'MSE': [lstm_metrics['MSE'], gru_metrics['MSE']],
        'RMSE': [lstm_metrics['RMSE'], gru_metrics['RMSE']],
        'MAE': [lstm_metrics['MAE'], gru_metrics['MAE']],
        'R²': [lstm_metrics['R2'], gru_metrics['R2']],
        'MAPE (%)': [lstm_metrics['MAPE'], gru_metrics['MAPE']]
    })
    
    print("\n")
    print(comparison_df.to_string(index=False))
    
    comparison_df.to_csv('../results/basic_model_comparison.csv', index=False)
    print("\n模型对比结果已保存为 '../results/basic_model_comparison.csv'")
    
    print("\n" + "=" * 70)
    print("任务完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
