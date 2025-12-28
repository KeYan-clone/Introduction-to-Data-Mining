"""
改进版时间序列预测任务：基于过去2小时的天气数据预测室外温度（OT）

改进内容：
1. 特征工程：添加时间特征、温度变化率、移动平均
2. 双向LSTM模型
3. 超参数调优
4. 性能对比分析
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
from datetime import datetime
from visualization_utils import plot_individual_charts, plot_comparison_charts

# 设置随机种子以保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')


def add_time_features(df):
    """
    添加时间特征工程
    
    Args:
        df: 包含date列的DataFrame
        
    Returns:
        添加了时间特征的DataFrame
    """
    print("\n添加时间特征...")
    
    # 解析日期时间
    df['datetime'] = pd.to_datetime(df['date'])
    
    # 提取时间特征
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['month'] = df['datetime'].dt.month
    
    # 时间的周期性编码（sin/cos变换）
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    print(f"添加了10个时间特征：hour, day_of_week, day_of_year, month, hour_sin/cos, day_sin/cos, month_sin/cos")
    
    return df


def add_statistical_features(df, target_col='OT', window_sizes=[3, 6, 12]):
    """
    添加统计特征（移动平均、变化率等）
    
    Args:
        df: DataFrame
        target_col: 目标列名
        window_sizes: 滑动窗口大小列表
        
    Returns:
        添加了统计特征的DataFrame
    """
    print("\n添加统计特征...")
    
    # 温度变化率
    df['temp_diff'] = df['T (degC)'].diff()
    df['temp_diff_2'] = df['T (degC)'].diff(2)
    
    # 目标变量的变化率
    df['ot_diff'] = df[target_col].diff()
    df['ot_diff_2'] = df[target_col].diff(2)
    
    # 移动平均
    for window in window_sizes:
        df[f'temp_ma_{window}'] = df['T (degC)'].rolling(window=window, min_periods=1).mean()
        df[f'ot_ma_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
        df[f'temp_std_{window}'] = df['T (degC)'].rolling(window=window, min_periods=1).std()
    
    # 填充初始的NaN值
    df.fillna(method='bfill', inplace=True)
    
    feature_count = 4 + len(window_sizes) * 3
    print(f"添加了{feature_count}个统计特征：温度变化率、移动平均、标准差")
    
    return df


def add_interaction_features(df):
    """
    添加特征交互项
    
    Args:
        df: DataFrame
        
    Returns:
        添加了交互特征的DataFrame
    """
    print("\n添加交互特征...")
    
    # 温度与湿度交互
    df['temp_humidity'] = df['T (degC)'] * df['rh (%)']
    
    # 温度与气压交互
    df['temp_pressure'] = df['T (degC)'] * df['p (mbar)']
    
    # 风速与温度交互
    df['wind_temp'] = df['wv (m/s)'] * df['T (degC)']
    
    print("添加了3个交互特征：temp_humidity, temp_pressure, wind_temp")
    
    return df


class WeatherDataset(Dataset):
    """时间序列数据集"""
    
    def __init__(self, data, window_size=12):
        self.data = data
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size, -1]  # OT是最后一列
        return torch.FloatTensor(x), torch.FloatTensor([y])


class BiLSTMPredictor(nn.Module):
    """基于双向LSTM的时间序列预测模型"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(BiLSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 双向
        )
        
        # 注意：双向LSTM输出维度是 hidden_size * 2
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        # BiLSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # 通过全连接层
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out


class LSTMPredictor(nn.Module):
    """优化的LSTM模型"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
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
    """优化的GRU模型"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(GRUPredictor, self).__init__()
        
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


def load_and_preprocess_data(file_path, window_size=12, train_ratio=0.8, use_feature_engineering=True):
    """
    加载并预处理数据（增强版）
    """
    print("=" * 50)
    print("1. 数据加载与预处理（增强版）")
    print("=" * 50)
    
    # 读取数据
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except:
            df = pd.read_csv(file_path, encoding='latin-1')
    
    print(f"原始数据形状: {df.shape}")
    
    if use_feature_engineering:
        # 特征工程
        df = add_time_features(df)
        df = add_statistical_features(df, target_col='OT')
        df = add_interaction_features(df)
    
    # 移除不需要的列
    columns_to_drop = ['date', 'datetime'] if 'datetime' in df.columns else ['date']
    features = df.drop(columns=columns_to_drop).values
    
    print(f"\n最终特征数据形状: {features.shape}")
    print(f"特征数量: {features.shape[1]}")
    
    # 检查NaN
    if np.isnan(features).any():
        print("警告：存在NaN值，进行填充...")
        features = np.nan_to_num(features, nan=0.0)
    
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
    
    # 创建数据集
    train_dataset = WeatherDataset(train_data, window_size)
    val_dataset = WeatherDataset(val_data, window_size)
    test_dataset = WeatherDataset(test_data, window_size)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, model_name='Model'):
    """
    训练模型（优化版）
    """
    print("\n" + "=" * 50)
    print(f"训练 {model_name}")
    print("=" * 50)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 15  # 增加patience
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
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'../models/{model_name.lower().replace(" ", "_")}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch+1}')
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(f'../models/{model_name.lower().replace(" ", "_")}.pth'))
    print(f'\n最佳验证损失: {best_val_loss:.6f}')
    
    return model, train_losses, val_losses


def evaluate_model(model, test_loader, scaler, model_name='Model'):
    """
    评估模型
    """
    print("\n" + "=" * 50)
    print(f"评估 {model_name}")
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


def generate_all_visualizations(results_dict):
    """
    为所有模型生成单独的图表和对比图表
    """
    print("\n" + "=" * 70)
    print("生成可视化图表")
    print("=" * 70)
    
    # 为每个模型生成7张单独的图表
    for model_name, data in results_dict.items():
        output_dir = f'../results/{model_name.lower().replace(" ", "_")}'
        
        print(f"\n正在为 {model_name} 生成图表...")
        plot_individual_charts(
            train_losses=data['train_losses'],
            val_losses=data['val_losses'],
            predictions=data['predictions'],
            actuals=data['actuals'],
            model_name=model_name,
            output_dir=output_dir
        )
    
    # 生成模型对比图表
    print("\n正在生成模型对比图表...")
    plot_comparison_charts(
        results_dict=results_dict,
        output_dir='../results'
    )
    
    print("\n所有可视化图表生成完成！")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("改进版时间序列预测：特征工程 + 超参数调优 + 双向LSTM")
    print("=" * 70)
    
    # 超参数配置
    WINDOW_SIZE = 12
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3  # 增加dropout
    NUM_EPOCHS = 60  # 增加训练轮数
    LEARNING_RATE = 0.0005  # 降低学习率
    
    data_path = '../data/weather.csv'
    if not os.path.exists(data_path):
        print(f"错误：数据文件 '{data_path}' 不存在！")
        return
    
    # 加载数据（使用特征工程）
    train_loader, val_loader, test_loader, scaler = load_and_preprocess_data(
        data_path, window_size=WINDOW_SIZE, use_feature_engineering=True
    )
    
    sample_x, _ = next(iter(train_loader))
    input_size = sample_x.shape[2]
    print(f"\n输入特征维度: {input_size}")
    
    results = {}
    
    # 1. 训练改进的LSTM
    print("\n" + "=" * 70)
    print("模型1: 改进的LSTM（特征工程+超参数调优）")
    print("=" * 70)
    
    lstm_model = LSTMPredictor(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    print(f"参数总数: {sum(p.numel() for p in lstm_model.parameters()):,}")
    
    lstm_model, lstm_train_losses, lstm_val_losses = train_model(
        lstm_model, train_loader, val_loader, 
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
        model_name='Improved LSTM'
    )
    
    lstm_predictions, lstm_actuals, lstm_metrics = evaluate_model(
        lstm_model, test_loader, scaler, model_name='Improved LSTM'
    )
    
    results['Improved LSTM'] = {
        'train_losses': lstm_train_losses,
        'val_losses': lstm_val_losses,
        'predictions': lstm_predictions,
        'actuals': lstm_actuals,
        'metrics': lstm_metrics
    }
    
    # 2. 训练双向LSTM
    print("\n" + "=" * 70)
    print("模型2: 双向LSTM（Bi-LSTM）")
    print("=" * 70)
    
    bilstm_model = BiLSTMPredictor(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    print(f"参数总数: {sum(p.numel() for p in bilstm_model.parameters()):,}")
    
    bilstm_model, bilstm_train_losses, bilstm_val_losses = train_model(
        bilstm_model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
        model_name='Bi-LSTM'
    )
    
    bilstm_predictions, bilstm_actuals, bilstm_metrics = evaluate_model(
        bilstm_model, test_loader, scaler, model_name='Bi-LSTM'
    )
    
    results['Bi-LSTM'] = {
        'train_losses': bilstm_train_losses,
        'val_losses': bilstm_val_losses,
        'predictions': bilstm_predictions,
        'actuals': bilstm_actuals,
        'metrics': bilstm_metrics
    }
    
    # 3. 训练改进的GRU（对比）
    print("\n" + "=" * 70)
    print("模型3: 改进的GRU（对比）")
    print("=" * 70)
    
    gru_model = GRUPredictor(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    print(f"参数总数: {sum(p.numel() for p in gru_model.parameters()):,}")
    
    gru_model, gru_train_losses, gru_val_losses = train_model(
        gru_model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
        model_name='Improved GRU'
    )
    
    gru_predictions, gru_actuals, gru_metrics = evaluate_model(
        gru_model, test_loader, scaler, model_name='Improved GRU'
    )
    
    results['Improved GRU'] = {
        'train_losses': gru_train_losses,
        'val_losses': gru_val_losses,
        'predictions': gru_predictions,
        'actuals': gru_actuals,
        'metrics': gru_metrics
    }
    
    # 生成所有可视化图表
    generate_all_visualizations(results)
    
    # 性能对比表
    print("\n" + "=" * 70)
    print("改进后的模型性能对比")
    print("=" * 70)
    
    comparison_data = []
    for name, data in results.items():
        comparison_data.append({
            'Model': name,
            'MSE': data['metrics']['MSE'],
            'RMSE': data['metrics']['RMSE'],
            'MAE': data['metrics']['MAE'],
            'R²': data['metrics']['R2'],
            'MAPE (%)': data['metrics']['MAPE']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n")
    print(comparison_df.to_string(index=False))
    
    # 保存结果
    os.makedirs('../results', exist_ok=True)
    comparison_df.to_csv('../results/improved_model_comparison.csv', index=False)
    print("\n对比结果已保存为 '../results/improved_model_comparison.csv'")
    
    # 找出最佳模型
    best_model = comparison_df.loc[comparison_df['R²'].idxmax()]
    print("\n" + "=" * 70)
    print("最佳模型")
    print("=" * 70)
    print(f"模型: {best_model['Model']}")
    print(f"R²: {best_model['R²']:.4f}")
    print(f"RMSE: {best_model['RMSE']:.4f}")
    print(f"MAE: {best_model['MAE']:.4f}")
    print(f"MAPE: {best_model['MAPE (%)']:.2f}%")
    
    print("\n" + "=" * 70)
    print("改进任务完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
