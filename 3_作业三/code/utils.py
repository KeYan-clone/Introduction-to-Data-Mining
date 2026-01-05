import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import torch
from torch.utils.data import Dataset

def set_seed(seed=42):
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_preprocess_data(data_path, date_col='date'):
    """Load CSV and basic preprocessing: parse date and fill NaNs."""
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(data_path, encoding='gbk')
        except:
            df = pd.read_csv(data_path, encoding='latin1')
    # replace common sentinel missing values (e.g. -9999) with NaN, then forward/back fill
    df.replace(-9999, np.nan, inplace=True)
    df.replace(-9999.0, np.nan, inplace=True)
    df[date_col] = pd.to_datetime(df[date_col], format='%Y/%m/%d %H:%M')
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def extract_time_features(df):
    df = df.copy()
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df

def create_sequences(data, seq_length=12):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, -1])
    return np.array(X), np.array(y)

def create_sequences_with_delta(data, seq_length=12):
    X, y_delta, y_last = [], [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        current_temp = data[i+seq_length-1, -1]
        next_temp = data[i+seq_length, -1]
        y_delta.append(next_temp - current_temp)
        y_last.append(current_temp)
    return np.array(X), np.array(y_delta), np.array(y_last)

def create_feature_target_sequences(features, target, seq_length=12):
    """Create sequences from feature matrix and target vector (no scaling).

    features: (N, F)
    target: (N,) or (N,1)
    returns X: (N-seq_length, seq_length, F), y: (N-seq_length,)
    """
    X, y = [], []
    tgt = target.reshape(-1)
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(tgt[i+seq_length])
    return np.array(X), np.array(y)

def create_feature_target_sequences_with_delta(features, target, seq_length=12):
    """Create sequences and return delta, last-value and next-value in raw space.

    returns X: (N-seq_length, seq_length, F), y_delta: (N-seq_length,), y_last: (N-seq_length,), y_next: (N-seq_length,)
    where y_delta = target[t+seq_length] - target[t+seq_length-1]
    y_last = target[t+seq_length-1] (OT at time t)
    y_next = target[t+seq_length] (OT at time t+1, ground truth)
    """
    X, y_delta, y_last, y_next = [], [], [], []
    tgt = target.reshape(-1)
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        last_temp = tgt[i+seq_length-1]  # OT_t
        next_temp = tgt[i+seq_length]     # OT_{t+1}
        y_delta.append(next_temp - last_temp)
        y_last.append(last_temp)
        y_next.append(next_temp)
    return np.array(X), np.array(y_delta), np.array(y_last), np.array(y_next)

class WeatherDataset(Dataset):
    def __init__(self, X, y, y_last=None, y_next=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.y_last = torch.FloatTensor(y_last) if y_last is not None else None
        self.y_next = torch.FloatTensor(y_next) if y_next is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y_last is None:
            return self.X[idx], self.y[idx]
        if self.y_next is None:
            return self.X[idx], self.y[idx], self.y_last[idx]
        return self.X[idx], self.y[idx], self.y_last[idx], self.y_next[idx]

def plot_predictions(actuals, predictions, model_name, save_path, num_points=500):
    plt.figure(figsize=(15, 6))
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

def plot_scatter(actuals, predictions, model_name, save_path):
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.5, s=10)
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

def generate_report(metrics_dict, model_name, report_path):
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# {model_name} 实验报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 性能指标\n\n")
        f.write("| 指标 | 值 |\n")
        f.write("|------|----|\n")
        for metric, value in metrics_dict.items():
            f.write(f"| {metric} | {value:.4f} |\n")
