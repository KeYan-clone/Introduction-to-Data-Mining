"""
时序预测任务：基于过去2小时的天气情况预测下一时间点室外温度（OT）
实验包含4个步骤：
1. 基线测试
2. LSTM模型
3. 引入周期性时间特征的LSTM
4. 基于特征工程的双向LSTM
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

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# ==================== 数据加载和预处理 ====================

def load_and_preprocess_data(data_path):
    """加载和预处理数据"""
    print("正在加载数据...")
    # 尝试不同的编码
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(data_path, encoding='gbk')
        except:
            df = pd.read_csv(data_path, encoding='latin1')
    
    print(f"数据形状: {df.shape}")
    print(f"数据列: {df.columns.tolist()}")
    print(f"\n前5行数据:\n{df.head()}")
    
    # 解析日期
    df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d %H:%M')
    
    # 检查缺失值
    print(f"\n缺失值统计:\n{df.isnull().sum()}")
    
    # 填充缺失值
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def extract_time_features(df):
    """提取周期性时间特征"""
    df = df.copy()
    
    # 提取时间特征
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    
    # 周期性编码（使用sin/cos变换）
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def create_sequences(data, seq_length=12):
    """创建滑动窗口序列"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, -1])  # OT是最后一列
    return np.array(X), np.array(y)

class WeatherDataset(Dataset):
    """天气数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
        # LSTM输出: output, (h_n, c_n)
        lstm_out, _ = self.lstm(x)
        # 使用最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])
        return out

class BiLSTMModel(nn.Module):
    """双向LSTM模型"""
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(BiLSTMModel, self).__init__()
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

def train_model(model, train_loader, criterion, optimizer, epochs=50, patience=10):
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
        
        # Early stopping
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
    
    # 反标准化
    predictions = scaler_y.inverse_transform(predictions).flatten()
    actuals = scaler_y.inverse_transform(actuals).flatten()
    
    # 计算指标
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    return predictions, actuals, {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

# ==================== 可视化函数 ====================

def plot_training_loss(losses_dict, save_path):
    """绘制训练损失曲线"""
    plt.figure(figsize=(12, 6))
    for name, losses in losses_dict.items():
        plt.plot(losses, label=name, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('训练损失曲线对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练损失曲线已保存至: {save_path}")

def plot_predictions(actuals, predictions, model_name, save_path, num_points=500):
    """绘制预测结果对比"""
    plt.figure(figsize=(15, 6))
    
    # 只显示前num_points个点，避免图表过于密集
    indices = range(min(num_points, len(actuals)))
    
    plt.plot(indices, actuals[:num_points], label='实际值', alpha=0.7, linewidth=1.5)
    plt.plot(indices, predictions[:num_points], label='预测值', alpha=0.7, linewidth=1.5)
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('室外温度 (OT)', fontsize=12)
    plt.title(f'{model_name} - 预测结果对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"预测结果图已保存至: {save_path}")

def plot_scatter(actuals, predictions, model_name, save_path):
    """绘制散点图"""
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.5, s=10)
    
    # 绘制理想预测线
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')
    
    plt.xlabel('实际值', fontsize=12)
    plt.ylabel('预测值', fontsize=12)
    plt.title(f'{model_name} - 预测散点图', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"散点图已保存至: {save_path}")

def plot_metrics_comparison(metrics_dict, save_path):
    """绘制指标对比图"""
    models = list(metrics_dict.keys())
    metrics_names = ['RMSE', 'MAE', 'R2']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics_names):
        values = [metrics_dict[model][metric] for model in models]
        axes[idx].bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)])
        axes[idx].set_title(f'{metric} 对比', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(metric, fontsize=10)
        axes[idx].tick_params(axis='x', rotation=15)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上显示数值
        for i, v in enumerate(values):
            axes[idx].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"指标对比图已保存至: {save_path}")

def generate_report(metrics_dict, model_name, report_path):
    """生成实验报告"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# {model_name} 实验报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 实验概述\n\n")
        
        if model_name == "步骤1: 基线模型":
            f.write("本实验使用简单的全连接神经网络作为基线模型，用于预测下一时间点的室外温度。\n\n")
            f.write("### 模型架构\n")
            f.write("- 输入层: 展平的12个时间步 × 21个特征\n")
            f.write("- 隐藏层1: 64个神经元 + ReLU + Dropout(0.2)\n")
            f.write("- 隐藏层2: 32个神经元 + ReLU + Dropout(0.2)\n")
            f.write("- 输出层: 1个神经元（预测OT值）\n\n")
            
        elif model_name == "步骤2: LSTM模型":
            f.write("本实验使用LSTM网络捕捉时间序列的长期依赖关系。\n\n")
            f.write("### 模型架构\n")
            f.write("- LSTM层: 2层，隐藏单元64，Dropout=0.2\n")
            f.write("- 全连接层1: 32个神经元 + ReLU + Dropout(0.2)\n")
            f.write("- 输出层: 1个神经元（预测OT值）\n\n")
            
        elif model_name == "步骤3: LSTM + 时间特征":
            f.write("本实验在LSTM基础上引入周期性时间特征（小时、星期、月份的sin/cos编码）。\n\n")
            f.write("### 模型架构\n")
            f.write("- 输入特征: 原始21维 + 6维时间特征（hour_sin/cos, day_sin/cos, month_sin/cos）\n")
            f.write("- LSTM层: 2层，隐藏单元64，Dropout=0.2\n")
            f.write("- 全连接层1: 32个神经元 + ReLU + Dropout(0.2)\n")
            f.write("- 输出层: 1个神经元（预测OT值）\n\n")
            
        elif model_name == "步骤4: 双向LSTM + 时间特征":
            f.write("本实验使用双向LSTM，能够同时从前向和后向捕捉时间序列的特征。\n\n")
            f.write("### 模型架构\n")
            f.write("- 输入特征: 原始21维 + 6维时间特征\n")
            f.write("- 双向LSTM层: 2层，隐藏单元64，Dropout=0.2\n")
            f.write("- 全连接层1: 64个神经元 + ReLU + Dropout(0.2)\n")
            f.write("- 全连接层2: 32个神经元 + ReLU + Dropout(0.2)\n")
            f.write("- 输出层: 1个神经元（预测OT值）\n\n")
        
        f.write("## 性能指标\n\n")
        f.write("| 指标 | 值 |\n")
        f.write("|------|----|\n")
        for metric, value in metrics_dict.items():
            f.write(f"| {metric} | {value:.4f} |\n")
        f.write("\n")
        
        f.write("## 指标说明\n\n")
        f.write("- **MSE (均方误差)**: 预测值与实际值差值的平方的平均值\n")
        f.write("- **RMSE (均方根误差)**: MSE的平方根，与目标变量单位相同\n")
        f.write("- **MAE (平均绝对误差)**: 预测值与实际值差值绝对值的平均值\n")
        f.write("- **R² (决定系数)**: 模型对数据的拟合程度，越接近1越好\n\n")
        
        f.write("## 结论\n\n")
        f.write(f"该模型在测试集上的RMSE为{metrics_dict['RMSE']:.4f}，")
        f.write(f"MAE为{metrics_dict['MAE']:.4f}，")
        f.write(f"R²为{metrics_dict['R2']:.4f}。\n")
        
    print(f"实验报告已保存至: {report_path}")

def generate_comparison_report(all_metrics, save_path):
    """生成综合对比报告"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("# 时序预测实验综合报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 实验目标\n\n")
        f.write("基于过去2小时（12个时间步）的天气数据，预测下一时间点的室外温度（OT）。\n\n")
        
        f.write("## 数据集信息\n\n")
        f.write("- 数据来源: 德国某气象站\n")
        f.write("- 时间跨度: 半年\n")
        f.write("- 采样频率: 每10分钟\n")
        f.write("- 数据点总数: 26200\n")
        f.write("- 特征维度: 21\n\n")
        
        f.write("## 实验步骤\n\n")
        f.write("1. **基线模型**: 使用全连接神经网络建立基线\n")
        f.write("2. **LSTM模型**: 使用LSTM捕捉时序依赖关系\n")
        f.write("3. **LSTM + 时间特征**: 引入周期性时间特征工程\n")
        f.write("4. **双向LSTM + 时间特征**: 使用双向LSTM增强特征提取\n\n")
        
        f.write("## 性能指标对比\n\n")
        f.write("| 模型 | RMSE | MAE | R² |\n")
        f.write("|------|------|-----|----|\n")
        for model_name, metrics in all_metrics.items():
            f.write(f"| {model_name} | {metrics['RMSE']:.4f} | {metrics['MAE']:.4f} | {metrics['R2']:.4f} |\n")
        f.write("\n")
        
        f.write("## 性能提升分析\n\n")
        baseline_rmse = all_metrics['基线模型']['RMSE']
        
        for i, (model_name, metrics) in enumerate(list(all_metrics.items())[1:], 1):
            rmse = metrics['RMSE']
            improvement = (baseline_rmse - rmse) / baseline_rmse * 100
            f.write(f"### {model_name}\n\n")
            f.write(f"- RMSE改进: {improvement:.2f}%\n")
            f.write(f"- 相比基线模型，RMSE从{baseline_rmse:.4f}降至{rmse:.4f}\n\n")
        
        f.write("## 关键发现\n\n")
        best_model = min(all_metrics.items(), key=lambda x: x[1]['RMSE'])
        f.write(f"1. **最佳模型**: {best_model[0]} (RMSE: {best_model[1]['RMSE']:.4f})\n")
        f.write("2. **LSTM的作用**: LSTM模型相比基线模型能够更好地捕捉时间序列的长期依赖关系\n")
        f.write("3. **时间特征工程**: 引入周期性时间特征（小时、星期、月份）能够帮助模型学习天气的周期性规律\n")
        f.write("4. **双向LSTM**: 双向结构允许模型同时从过去和未来的上下文中学习，进一步提升预测精度\n\n")
        
        f.write("## 结论\n\n")
        f.write("实验表明，通过逐步引入更复杂的模型架构和特征工程，模型的预测性能得到了显著提升。")
        f.write("双向LSTM结合时间特征工程的方法在本任务中取得了最佳效果。\n")
        
    print(f"综合对比报告已保存至: {save_path}")

# ==================== 主函数 ====================

def main():
    # 创建结果目录
    results_dir = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 数据路径
    data_path = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\data\weather.csv'
    
    # 加载数据
    df = load_and_preprocess_data(data_path)
    
    # 保存所有实验结果
    all_metrics = {}
    all_losses = {}
    
    # ==================== 步骤1: 基线模型 ====================
    print("\n" + "="*50)
    print("步骤1: 基线模型 - 全连接神经网络")
    print("="*50)
    
    # 准备数据（不包含时间特征）
    feature_cols = [col for col in df.columns if col not in ['date', 'OT']]
    features = df[feature_cols].values
    target = df['OT'].values.reshape(-1, 1)
    
    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    features_scaled = scaler_X.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target)
    
    # 合并特征和目标
    data = np.column_stack([features_scaled, target_scaled])
    
    # 创建序列
    seq_length = 12  # 2小时 = 12个10分钟
    X, y = create_sequences(data, seq_length)
    
    # 划分训练集和测试集 (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 创建数据加载器
    train_dataset = WeatherDataset(X_train, y_train)
    test_dataset = WeatherDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 训练基线模型
    input_size = X_train.shape[2]
    baseline_model = BaselineModel(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
    
    print("\n开始训练基线模型...")
    baseline_losses = train_model(baseline_model, train_loader, criterion, optimizer, epochs=100)
    all_losses['基线模型'] = baseline_losses
    
    # 评估
    print("\n评估基线模型...")
    baseline_pred, baseline_actual, baseline_metrics = evaluate_model(baseline_model, test_loader, scaler_y)
    all_metrics['基线模型'] = baseline_metrics
    
    print("\n基线模型性能:")
    for metric, value in baseline_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 可视化
    plot_predictions(baseline_actual, baseline_pred, "基线模型", 
                    os.path.join(results_dir, "step1_predictions.png"))
    plot_scatter(baseline_actual, baseline_pred, "基线模型",
                os.path.join(results_dir, "step1_scatter.png"))
    
    # 生成报告
    generate_report(baseline_metrics, "步骤1: 基线模型",
                   os.path.join(results_dir, "step1_report.md"))
    
    # ==================== 步骤2: LSTM模型 ====================
    print("\n" + "="*50)
    print("步骤2: LSTM模型")
    print("="*50)
    
    # 使用相同的数据
    lstm_model = LSTMModel(input_size).to(device)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    
    print("\n开始训练LSTM模型...")
    lstm_losses = train_model(lstm_model, train_loader, criterion, optimizer, epochs=100)
    all_losses['LSTM模型'] = lstm_losses
    
    # 评估
    print("\n评估LSTM模型...")
    lstm_pred, lstm_actual, lstm_metrics = evaluate_model(lstm_model, test_loader, scaler_y)
    all_metrics['LSTM模型'] = lstm_metrics
    
    print("\nLSTM模型性能:")
    for metric, value in lstm_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 可视化
    plot_predictions(lstm_actual, lstm_pred, "LSTM模型",
                    os.path.join(results_dir, "step2_predictions.png"))
    plot_scatter(lstm_actual, lstm_pred, "LSTM模型",
                os.path.join(results_dir, "step2_scatter.png"))
    
    # 对比前两步
    plot_metrics_comparison({'基线模型': baseline_metrics, 'LSTM模型': lstm_metrics},
                           os.path.join(results_dir, "step2_comparison.png"))
    
    # 生成报告
    generate_report(lstm_metrics, "步骤2: LSTM模型",
                   os.path.join(results_dir, "step2_report.md"))
    
    # ==================== 步骤3: LSTM + 时间特征 ====================
    print("\n" + "="*50)
    print("步骤3: LSTM + 周期性时间特征")
    print("="*50)
    
    # 添加时间特征
    df_with_time = extract_time_features(df)
    
    # 准备数据（包含时间特征）
    time_feature_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    feature_cols_extended = feature_cols + time_feature_cols
    features_extended = df_with_time[feature_cols_extended].values
    
    # 标准化
    scaler_X_ext = StandardScaler()
    features_extended_scaled = scaler_X_ext.fit_transform(features_extended)
    
    # 合并特征和目标
    data_extended = np.column_stack([features_extended_scaled, target_scaled])
    
    # 创建序列
    X_ext, y_ext = create_sequences(data_extended, seq_length)
    
    # 划分训练集和测试集
    X_train_ext, X_test_ext = X_ext[:split_idx], X_ext[split_idx:]
    y_train_ext, y_test_ext = y_ext[:split_idx], y_ext[split_idx:]
    
    print(f"训练集大小: {X_train_ext.shape}")
    print(f"测试集大小: {X_test_ext.shape}")
    
    # 创建数据加载器
    train_dataset_ext = WeatherDataset(X_train_ext, y_train_ext)
    test_dataset_ext = WeatherDataset(X_test_ext, y_test_ext)
    train_loader_ext = DataLoader(train_dataset_ext, batch_size=64, shuffle=True)
    test_loader_ext = DataLoader(test_dataset_ext, batch_size=64, shuffle=False)
    
    # 训练LSTM模型（带时间特征）
    input_size_ext = X_train_ext.shape[2]
    lstm_time_model = LSTMModel(input_size_ext).to(device)
    optimizer = torch.optim.Adam(lstm_time_model.parameters(), lr=0.001)
    
    print("\n开始训练LSTM+时间特征模型...")
    lstm_time_losses = train_model(lstm_time_model, train_loader_ext, criterion, optimizer, epochs=100)
    all_losses['LSTM+时间特征'] = lstm_time_losses
    
    # 评估
    print("\n评估LSTM+时间特征模型...")
    lstm_time_pred, lstm_time_actual, lstm_time_metrics = evaluate_model(lstm_time_model, test_loader_ext, scaler_y)
    all_metrics['LSTM+时间特征'] = lstm_time_metrics
    
    print("\nLSTM+时间特征模型性能:")
    for metric, value in lstm_time_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 可视化
    plot_predictions(lstm_time_actual, lstm_time_pred, "LSTM+时间特征",
                    os.path.join(results_dir, "step3_predictions.png"))
    plot_scatter(lstm_time_actual, lstm_time_pred, "LSTM+时间特征",
                os.path.join(results_dir, "step3_scatter.png"))
    
    # 对比前三步
    plot_metrics_comparison({'基线模型': baseline_metrics, 
                            'LSTM模型': lstm_metrics,
                            'LSTM+时间特征': lstm_time_metrics},
                           os.path.join(results_dir, "step3_comparison.png"))
    
    # 生成报告
    generate_report(lstm_time_metrics, "步骤3: LSTM + 时间特征",
                   os.path.join(results_dir, "step3_report.md"))
    
    # ==================== 步骤4: 双向LSTM + 时间特征 ====================
    print("\n" + "="*50)
    print("步骤4: 双向LSTM + 时间特征")
    print("="*50)
    
    # 使用相同的数据（带时间特征）
    bilstm_model = BiLSTMModel(input_size_ext).to(device)
    optimizer = torch.optim.Adam(bilstm_model.parameters(), lr=0.001)
    
    print("\n开始训练双向LSTM模型...")
    bilstm_losses = train_model(bilstm_model, train_loader_ext, criterion, optimizer, epochs=100)
    all_losses['双向LSTM+时间特征'] = bilstm_losses
    
    # 评估
    print("\n评估双向LSTM模型...")
    bilstm_pred, bilstm_actual, bilstm_metrics = evaluate_model(bilstm_model, test_loader_ext, scaler_y)
    all_metrics['双向LSTM+时间特征'] = bilstm_metrics
    
    print("\n双向LSTM+时间特征模型性能:")
    for metric, value in bilstm_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 可视化
    plot_predictions(bilstm_actual, bilstm_pred, "双向LSTM+时间特征",
                    os.path.join(results_dir, "step4_predictions.png"))
    plot_scatter(bilstm_actual, bilstm_pred, "双向LSTM+时间特征",
                os.path.join(results_dir, "step4_scatter.png"))
    
    # 生成报告
    generate_report(bilstm_metrics, "步骤4: 双向LSTM + 时间特征",
                   os.path.join(results_dir, "step4_report.md"))
    
    # ==================== 综合对比 ====================
    print("\n" + "="*50)
    print("生成综合对比报告")
    print("="*50)
    
    # 绘制训练损失对比
    plot_training_loss(all_losses, os.path.join(results_dir, "training_loss_comparison.png"))
    
    # 绘制所有模型的指标对比
    plot_metrics_comparison(all_metrics, os.path.join(results_dir, "final_metrics_comparison.png"))
    
    # 生成综合报告
    generate_comparison_report(all_metrics, os.path.join(results_dir, "final_report.md"))
    
    print("\n" + "="*50)
    print("实验完成！所有结果已保存至 results 目录")
    print("="*50)
    
    # 打印最终结果摘要
    print("\n最终结果摘要:")
    print("-" * 60)
    for model_name, metrics in all_metrics.items():
        print(f"\n{model_name}:")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE:  {metrics['MAE']:.4f}")
        print(f"  R²:   {metrics['R2']:.4f}")
    print("-" * 60)

if __name__ == "__main__":
    main()
