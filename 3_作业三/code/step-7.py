"""
步骤7: 双向LSTM + 全面特征工程

实验内容：
- 在步骤4的基础上添加全面的特征工程
- 衍生气象特征：露点、相对湿度到绝对湿度、风速分量、温差、气压变化量
- 多尺度滞后与滑动统计：rolling mean/std/min/max/slope
- 周期性时间编码：扩展的时间特征
- 太阳位置估计：白天/夜间标识
- 差分特征：一阶和二阶差分
- 模型架构：BiLSTM(2层,64隐藏单元) -> 全连接层(64) -> 全连接层(32) -> 输出层(1)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import signal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from utils import set_seed, get_device, load_and_preprocess_data, extract_time_features, create_feature_target_sequences, WeatherDataset, plot_predictions, plot_scatter

set_seed(42)
device = get_device()
print(f'使用设备: {device}')

# ==================== 特征工程函数 ====================

def calculate_dew_point(temp, rh):
    """
    计算露点温度
    使用Magnus-Tetens公式
    """
    a = 17.27
    b = 237.7
    alpha = ((a * temp) / (b + temp)) + np.log(rh / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    return dew_point

def calculate_absolute_humidity(temp, rh):
    """
    计算绝对湿度（g/m³）
    """
    # 饱和水蒸气压 (hPa)
    es = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))
    # 实际水蒸气压
    e = es * (rh / 100.0)
    # 绝对湿度
    abs_humidity = (e * 2.1674) / (273.15 + temp)
    return abs_humidity

def calculate_wind_components(ws, wd):
    """
    计算风的u、v分量
    u: 东西分量 (正向东)
    v: 南北分量 (正向北)
    """
    wd_rad = np.deg2rad(wd)
    u = -ws * np.sin(wd_rad)
    v = -ws * np.cos(wd_rad)
    return u, v

def calculate_pressure_change(pressure, window=6):
    """
    计算气压变化量（使用历史数据，避免泄露）
    """
    pressure_change = pressure.diff(periods=1).fillna(0)
    # 使用shift确保只使用历史数据
    pressure_change_ma = pressure_change.rolling(window=window, min_periods=1).mean().shift(1)
    return pressure_change, pressure_change_ma.fillna(0)

def calculate_rolling_slope(series, window):
    """
    计算滑动窗口内的线性斜率
    """
    def linear_slope(y):
        if len(y) < 2:
            return 0
        x = np.arange(len(y))
        try:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        except:
            return 0
    
    # 使用shift确保只使用历史数据
    slopes = series.rolling(window=window, min_periods=2).apply(linear_slope, raw=True).shift(1)
    return slopes.fillna(0)

def calculate_solar_position(df):
    """
    估算太阳位置特征
    简化模型：基于时间判断白天/夜间
    德国纬度约50°N，夏至日出约5:00，日落约21:00；冬至日出约8:00，日落约16:00
    """
    df = df.copy()
    
    # 从date列提取时间信息
    if 'date' in df.columns:
        hour = df['date'].dt.hour
        month = df['date'].dt.month
    else:
        # 如果没有date列，使用已有的hour和month列
        hour = df['hour'] if 'hour' in df.columns else 12
        month = df['month'] if 'month' in df.columns else 6
    
    # 估算日出日落时间（随月份变化）
    # 1月: 8:00-16:00, 7月: 5:00-21:00
    sunrise_hour = 8 - (month - 1) * 3 / 6  # 冬季8点，夏季5点
    sunset_hour = 16 + (month - 1) * 5 / 6   # 冬季16点，夏季21点
    
    # 白天标识
    df['is_daytime'] = ((hour >= sunrise_hour) & (hour <= sunset_hour)).astype(int)
    
    # 太阳高度角估计（简化：正弦函数）
    hour_from_noon = hour - 12
    df['solar_elevation'] = np.maximum(0, np.cos(hour_from_noon * np.pi / 12) * 
                                       (1 + 0.3 * np.sin(month * np.pi / 6)))
    
    return df

def add_derived_features(df):
    """
    添加衍生气象特征
    """
    df = df.copy()
    
    # 1. 露点温度
    df['dew_point'] = calculate_dew_point(df['T (degC)'], df['rh (%)'])
    
    # 2. 绝对湿度
    df['abs_humidity'] = calculate_absolute_humidity(df['T (degC)'], df['rh (%)'])
    
    # 3. 风速分量
    df['wind_u'], df['wind_v'] = calculate_wind_components(df['wv (m/s)'], df['wd (deg)'])
    df['wind_speed_squared'] = df['wv (m/s)'] ** 2
    
    # 4. 温差特征（OT与其他温度传感器）
    # 注意：这里使用历史数据，在后续滑动窗口中只能看到过去的温差
    df['temp_diff_T'] = df['T (degC)'] - df['Tpot (K)'] + 273.15
    df['temp_diff_Tdew'] = df['T (degC)'] - df['Tdew (degC)']
    df['temp_diff_Tlog'] = df['T (degC)'] - df['Tlog (degC)']
    
    # 5. 气压变化量
    df['pressure_change'], df['pressure_change_ma'] = calculate_pressure_change(df['p (mbar)'])
    
    return df

def add_rolling_statistics(df, windows=[6, 12, 36, 72]):
    """
    添加多尺度滚动统计特征（只针对非目标变量，避免泄露）
    windows: [6(1h), 12(2h), 36(6h), 72(12h)]
    """
    df = df.copy()
    
    # 选择关键特征进行统计（避免OT）
    key_features = ['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)', 
                    'Tdew (degC)', 'sh (g/kg)']
    
    for feature in key_features:
        if feature not in df.columns:
            continue
            
        feature_short = feature.split('(')[0].strip().replace(' ', '_')
        
        for window in windows:
            # 均值
            df[f'{feature_short}_ma_{window}'] = df[feature].rolling(
                window=window, min_periods=1).mean().shift(1).fillna(method='bfill')
            
            # 标准差
            df[f'{feature_short}_std_{window}'] = df[feature].rolling(
                window=window, min_periods=1).std().shift(1).fillna(0)
            
            # 最小值
            df[f'{feature_short}_min_{window}'] = df[feature].rolling(
                window=window, min_periods=1).min().shift(1).fillna(method='bfill')
            
            # 最大值
            df[f'{feature_short}_max_{window}'] = df[feature].rolling(
                window=window, min_periods=1).max().shift(1).fillna(method='bfill')
            
            # 线性斜率
            df[f'{feature_short}_slope_{window}'] = calculate_rolling_slope(df[feature], window)
    
    return df

def add_difference_features(df):
    """
    添加差分特征（针对关键变量）
    """
    df = df.copy()
    
    key_features = ['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)']
    
    for feature in key_features:
        if feature not in df.columns:
            continue
            
        feature_short = feature.split('(')[0].strip().replace(' ', '_')
        
        # 一阶差分（使用shift避免泄露）
        df[f'{feature_short}_diff1'] = df[feature].diff(1).shift(1).fillna(0)
        
        # 二阶差分
        df[f'{feature_short}_diff2'] = df[feature].diff(2).shift(1).fillna(0)
    
    return df

def add_interaction_features(df):
    """
    添加交互特征
    """
    df = df.copy()
    
    # 温度与湿度交互
    df['temp_humidity_interaction'] = df['T (degC)'] * df['rh (%)']
    
    # 气压与温度交互
    df['pressure_temp_interaction'] = df['p (mbar)'] * df['T (degC)']
    
    # 风速与温度交互
    df['wind_temp_interaction'] = df['wv (m/s)'] * df['T (degC)']
    
    return df

def add_all_features(df):
    """
    添加所有特征工程
    """
    print("开始特征工程...")
    
    # 1. 时间特征（扩展版）
    df = extract_time_features(df)
    print(f"  - 添加时间特征后: {df.shape[1]} 列")
    
    # 2. 太阳位置
    df = calculate_solar_position(df)
    print(f"  - 添加太阳位置特征后: {df.shape[1]} 列")
    
    # 3. 衍生气象特征
    df = add_derived_features(df)
    print(f"  - 添加衍生气象特征后: {df.shape[1]} 列")
    
    # 4. 交互特征
    df = add_interaction_features(df)
    print(f"  - 添加交互特征后: {df.shape[1]} 列")
    
    # 5. 差分特征
    df = add_difference_features(df)
    print(f"  - 添加差分特征后: {df.shape[1]} 列")
    
    # 6. 滚动统计特征
    df = add_rolling_statistics(df)
    print(f"  - 添加滚动统计特征后: {df.shape[1]} 列")
    
    # 填充任何剩余的NaN
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    print(f"特征工程完成！总特征数: {df.shape[1]}")
    
    return df

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

def train_model(model, train_loader, criterion, optimizer, epochs=100, patience=15):
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
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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

def generate_report(metrics, title, save_path):
    """生成实验报告"""
    report = f"""# {title}

## 模型性能指标

| 指标 | 值 |
|------|-----|
| MSE  | {metrics['MSE']:.4f} |
| RMSE | {metrics['RMSE']:.4f} |
| MAE  | {metrics['MAE']:.4f} |
| R²   | {metrics['R2']:.4f} |

## 实验配置

- 序列长度: 12个时间点（2小时）
- 模型: 双向LSTM (2层, 64隐藏单元)
- 优化器: Adam (lr=0.001)
- 批大小: 64
- 训练/测试分割: 80%/20%

## 特征工程

本实验在步骤4的基础上添加了全面的特征工程：

1. **衍生气象特征**：
   - 露点温度
   - 绝对湿度
   - 风速分量 (u, v)
   - 风速平方
   - 温差特征
   - 气压变化量

2. **多尺度滚动统计**（1h, 2h, 6h, 12h）：
   - 滚动均值
   - 滚动标准差
   - 滚动最小/最大值
   - 滚动线性斜率

3. **周期性时间特征**：
   - 小时/天/月的sin/cos编码
   - 白天/夜间标识
   - 太阳高度角估计

4. **差分特征**：
   - 一阶差分
   - 二阶差分

5. **交互特征**：
   - 温度×湿度
   - 气压×温度
   - 风速×温度

## 结论

通过全面的特征工程，模型能够更好地捕捉气象数据中的复杂模式和关系。
"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"报告已保存至: {save_path}")

# ==================== 主函数 ====================

def main():
    print("="*50)
    print("步骤7: 双向LSTM + 全面特征工程")
    print("="*50)
    
    results_dir = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\results'
    docs_dir = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\docs'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    data_path = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\data\weather.csv'
    
    # 加载数据
    df = load_and_preprocess_data(data_path)
    print(f"原始数据形状: {df.shape}")
    
    # 添加所有特征
    df_featured = add_all_features(df)
    
    # 准备特征和目标
    feature_cols = [col for col in df_featured.columns if col not in ['date', 'OT']]
    print(f"\n总特征数: {len(feature_cols)}")
    print(f"特征列表（前20个）: {feature_cols[:20]}")
    
    features = df_featured[feature_cols].values
    target = df_featured['OT'].values.reshape(-1, 1)
    
    # 创建序列
    seq_length = 12
    X, y = create_feature_target_sequences(features, target, seq_length)
    print(f"\n序列形状: X={X.shape}, y={y.shape}")
    
    # 划分训练集和测试集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 标准化
    n_features = X_train.shape[2]
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    scaler_X.fit(X_train_flat)
    X_train_scaled = scaler_X.transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test.shape)
    
    scaler_y.fit(y_train.reshape(-1, 1))
    y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 创建数据加载器
    train_dataset = WeatherDataset(X_train_scaled, y_train_scaled)
    test_dataset = WeatherDataset(X_test_scaled, y_test_scaled)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 创建模型
    input_size = X_train.shape[2]
    model = BiLSTMModel(input_size, hidden_size=64, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n模型输入维度: {input_size}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    print("\n开始训练...")
    train_losses = train_model(model, train_loader, criterion, optimizer, epochs=100, patience=15)
    
    # 评估模型
    print("\n评估模型...")
    predictions, actuals, metrics = evaluate_model(model, test_loader, scaler_y)
    
    # 打印结果
    print("\n" + "="*50)
    print("模型性能:")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 可视化
    plot_predictions(actuals, predictions, "双向LSTM+全面特征工程", 
                    os.path.join(results_dir, "step7_predictions.png"))
    plot_scatter(actuals, predictions, "双向LSTM+全面特征工程",
                os.path.join(results_dir, "step7_scatter.png"))
    
    # 生成报告
    generate_report(metrics, "步骤7: 双向LSTM + 全面特征工程",
                   os.path.join(docs_dir, "step7_report.md"))
    
    print("\n实验完成！")
    print(f"结果已保存至: {results_dir}")
    print(f"报告已保存至: {docs_dir}")

if __name__ == '__main__':
    main()
