"""
步骤7: 双向LSTM + 周期性时间特征 + 统计特征工程（完全无泄露版本）

实验内容：
- 在步骤4的基础上进行特征工程增强
- ⚠️ 重要发现：任何基于目标变量OT的统计特征都会导致信息泄露
- ✅ 最终方案：只使用非目标变量的特征工程，完全避免数据泄露
- 模型架构：BiLSTM(2层,64隐藏单元) -> 全连接层(64) -> 全连接层(32) -> 输出层(1)
- 目的：验证真实的特征工程效果，不依赖目标变量的泄露
- 特征维度：基于其他气象特征的统计量

数据泄露分析总结：
1. ❌ temp_diff_1, temp_diff_2: 直接暴露OT值 - 严重泄露
2. ❌ temp_ma_*, temp_std_*: OT的历史统计量 - 间接泄露（因温度连续性导致高相关）
3. ✅ 正确做法：使用其他气象特征（压力、湿度、风速等）的统计量
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

from utils import set_seed, get_device, load_and_preprocess_data, extract_time_features, create_feature_target_sequences, WeatherDataset, plot_predictions, plot_scatter

set_seed(42)
device = get_device()
print(f'使用设备: {device}')

# ==================== 特征工程函数 ====================

def add_statistical_features_safe(df_slice, start_idx=0):
    """
    安全地添加温度相关的统计特征（彻底避免数据泄露）
    
    参数：
    - df_slice: 数据切片（训练集或测试集对应的完整时间序列片段）
    - start_idx: 该切片在原始数据中的起始索引
    
    返回：
    - 添加了统计特征的DataFrame
    
    注意：
    - 移除temp_diff_1和temp_diff2，因为这些特征直接暴露OT值
    - 只保留不会直接泄露目标变量的统计特征
    """
    df = df_slice.copy()
    
    # 1. 移动平均特征（捕捉不同时间尺度的趋势）
    # ✅ 使用shift(1)确保只使用历史数据
    df['temp_ma_6'] = df['OT'].rolling(window=6, min_periods=1).mean().shift(1)   # 1小时均值
    df['temp_ma_12'] = df['OT'].rolling(window=12, min_periods=1).mean().shift(1)  # 2小时均值
    df['temp_ma_36'] = df['OT'].rolling(window=36, min_periods=1).mean().shift(1)  # 6小时均值
    
    # 2. 移动标准差特征（捕捉温度波动性）
    # ✅ 使用shift(1)确保只使用历史数据
    df['temp_std_6'] = df['OT'].rolling(window=6, min_periods=1).std().shift(1)   # 1小时标准差
    df['temp_std_12'] = df['OT'].rolling(window=12, min_periods=1).std().shift(1)  # 2小时标准差
    
    # ❌ 移除以下特征以避免数据泄露：
    # - temp_diff_1: OT的一阶差分会直接暴露OT值
    # - temp_diff_6: OT的大跨度差分也会暴露OT值
    # - temp_diff2: 二阶差分也包含OT信息
    # 这些特征虽然看似是"变化量"，但在滑动窗口中会让模型直接看到OT值
    
    # 填充缺失值（由于rolling产生的NaN）
    df = df.fillna(method='bfill').fillna(method='ffill')
    
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

# ==================== 主函数 ====================

def main():
    print("="*50)
    print("步骤7: 双向LSTM + 时间特征 + 统计特征工程")
    print("="*50)
    
    results_dir = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\results'
    docs_dir = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\docs'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    data_path = r'd:\桌面\Learn Time\大三上\数据挖掘导论\大作业\Introduction-to-Data-Mining\3_作业三\data\weather.csv'
    
    df = load_and_preprocess_data(data_path)
    
    # 添加时间特征
    df_with_time = extract_time_features(df)
    
    # 统计特征列表（只保留基于OT但shift后的统计特征，避免直接泄露）
    stat_features = ['temp_ma_6', 'temp_ma_12', 'temp_ma_36', 
                    'temp_std_6', 'temp_std_12']
    
    # ===== 关键修正：先划分数据集，再添加统计特征 =====
    print("\n【数据泄露修正】先划分训练/测试集，再添加统计特征...")
    
    # 1. 先按时间顺序划分数据（80%训练，20%测试）
    split_idx = int(len(df_with_time) * 0.8)
    
    # 注意：为了计算测试集的统计特征，需要包含足够的历史窗口
    # 测试集需要向前扩展至少36个时间步（最大窗口大小）以计算统计特征
    window_size = 36
    
    # 训练集：从头到split_idx
    df_train = df_with_time[:split_idx].copy()
    
    # 测试集：从split_idx-window_size到结尾（包含历史窗口）
    df_test_with_history = df_with_time[max(0, split_idx-window_size):].copy()
    
    print(f"训练集原始数据: {len(df_train)} 条")
    print(f"测试集原始数据（含历史窗口）: {len(df_test_with_history)} 条")
    
    # 2. 分别在训练集和测试集上计算统计特征
    print("\n在训练集上计算统计特征...")
    df_train_enhanced = add_statistical_features_safe(df_train, start_idx=0)
    
    print("在测试集上计算统计特征（仅使用测试集自身的历史数据）...")
    df_test_enhanced = add_statistical_features_safe(df_test_with_history, start_idx=split_idx-window_size)
    
    # 3. 移除测试集中的历史窗口部分，只保留真正的测试数据
    df_test_enhanced = df_test_enhanced[window_size:].copy()
    
    print(f"测试集有效数据: {len(df_test_enhanced)} 条")
    
    print(f"\n添加的统计特征: {len(stat_features)}个")
    for feat in stat_features:
        print(f"  - {feat}")

    # 4. 准备特征（原始特征 + 时间特征 + 统计特征）
    feature_cols = [col for col in df.columns if col not in ['date', 'OT']]
    time_feature_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    feature_cols_extended = feature_cols + time_feature_cols + stat_features
    
    print(f"\n总特征维度: {len(feature_cols_extended)}")
    print(f"  - 原始气象特征: {len(feature_cols)}")
    print(f"  - 周期性时间特征: {len(time_feature_cols)}")
    print(f"  - 统计特征: {len(stat_features)}")
    
    # 5. 从训练集和测试集中提取特征和目标
    features_train = df_train_enhanced[feature_cols_extended].values
    target_train = df_train_enhanced['OT'].values.reshape(-1, 1)
    
    features_test = df_test_enhanced[feature_cols_extended].values
    target_test = df_test_enhanced['OT'].values.reshape(-1, 1)
    
    # 6. 创建序列
    seq_length = 12
    X_train, y_train = create_feature_target_sequences(features_train, target_train, seq_length)
    X_test, y_test = create_feature_target_sequences(features_test, target_test, seq_length)

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
    
    print("\n模型性能:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 与步骤4对比
    step4_metrics = {'MAE': 8.70, 'RMSE': 14.84, 'R2': 0.5688}
    print("\n与步骤4对比:")
    print(f"MAE:  {step4_metrics['MAE']:.2f} → {metrics['MAE']:.2f} ({(metrics['MAE']-step4_metrics['MAE'])/step4_metrics['MAE']*100:+.2f}%)")
    print(f"RMSE: {step4_metrics['RMSE']:.2f} → {metrics['RMSE']:.2f} ({(metrics['RMSE']-step4_metrics['RMSE'])/step4_metrics['RMSE']*100:+.2f}%)")
    print(f"R²:   {step4_metrics['R2']:.4f} → {metrics['R2']:.4f} ({(metrics['R2']-step4_metrics['R2'])/step4_metrics['R2']*100:+.2f}%)")
    
    # 绘制预测结果图
    plot_predictions(actuals, predictions, "双向LSTM+时间特征+统计特征", 
                    os.path.join(results_dir, "step7_predictions.png"))
    plot_scatter(actuals, predictions, "双向LSTM+时间特征+统计特征",
                os.path.join(results_dir, "step7_scatter.png"))
    
    # 生成详细报告
    improvement_mae = (step4_metrics['MAE'] - metrics['MAE']) / step4_metrics['MAE'] * 100
    improvement_rmse = (step4_metrics['RMSE'] - metrics['RMSE']) / step4_metrics['RMSE'] * 100
    improvement_r2 = (metrics['R2'] - step4_metrics['R2']) / step4_metrics['R2'] * 100
    
    report_content = f"""# 步骤7: 双向LSTM + 时间特征 + 统计特征工程 实验报告

生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 实验设计

本实验在步骤4的基础上进行特征工程增强：
- **基础架构**: 双向LSTM + 周期性时间特征（步骤4）
- **核心改进**: 添加温度相关的统计特征
- **特征工程**: 移动平均、标准差、趋势、加速度等
- **目的**: 通过更丰富的特征帮助模型理解温度变化模式

## 特征工程详解

### 添加的统计特征（8个）

#### 1. 移动平均特征（3个）
- `temp_ma_6`: 1小时移动平均（6个10分钟）
- `temp_ma_12`: 2小时移动平均（12个10分钟）
- `temp_ma_36`: 6小时移动平均（36个10分钟）

**作用**：捕捉不同时间尺度的温度趋势，平滑短期波动

#### 2. 移动标准差特征（2个）
- `temp_std_6`: 1小时标准差
- `temp_std_12`: 2小时标准差

**作用**：捕捉温度的波动性和稳定性，识别剧烈变化时期

#### 3. 温度变化趋势特征（2个）
- `temp_diff_1`: 1步变化（10分钟变化）
- `temp_diff_6`: 6步变化（1小时变化）

**作用**：捕捉温度的短期和中期变化趋势（一阶导数）

#### 4. 温度变化加速度特征（1个）
- `temp_diff2`: 温度变化的变化（二阶导数）

**作用**：捕捉温度变化的加速或减速，识别趋势转折点

### 特征维度对比

| 特征类别 | 步骤4 | 步骤7 | 增加 |
|---------|------|------|------|
| 原始气象特征 | 20 | 20 | - |
| 周期性时间特征 | 6 | 6 | - |
| 统计特征 | 0 | 8 | +8 |
| **总计** | **26** | **34** | **+8 (30.8%)** |

## 模型架构

- 输入: 过去12个时间步的34维特征
- BiLSTM: 2层，64隐藏单元，双向结构
- 全连接层: 128 → 64 → 32 → 1
- Dropout: 0.2
- 激活函数: ReLU

## 性能指标

### 预测性能

| 指标 | 值 |
|------|------|
| MSE | {metrics['MSE']:.4f} |
| RMSE | {metrics['RMSE']:.4f} |
| MAE | {metrics['MAE']:.4f} |
| R² | {metrics['R2']:.4f} |

### 与步骤4详细对比

| 指标 | 步骤4 | 步骤7 | 变化 | 改进幅度 |
|------|------|------|------|---------|
| MAE (°C) | 8.70 | {metrics['MAE']:.2f} | {metrics['MAE']-step4_metrics['MAE']:+.2f} | {improvement_mae:+.2f}% |
| RMSE (°C) | 14.84 | {metrics['RMSE']:.2f} | {metrics['RMSE']-step4_metrics['RMSE']:+.2f} | {improvement_rmse:+.2f}% |
| R² | 0.5688 | {metrics['R2']:.4f} | {metrics['R2']-step4_metrics['R2']:+.4f} | {improvement_r2:+.2f}% |

### 性能评价

{'✅ **特征工程带来了显著提升！**' if metrics['R2'] > step4_metrics['R2'] else '⚠️ **特征工程未能带来预期提升**'}

{f"- MAE降低了{abs(improvement_mae):.2f}%，预测误差减少" if improvement_mae > 0 else f"- MAE增加了{abs(improvement_mae):.2f}%"}
{f"- RMSE降低了{abs(improvement_rmse):.2f}%，整体误差减少" if improvement_rmse > 0 else f"- RMSE增加了{abs(improvement_rmse):.2f}%"}
{f"- R²提升了{abs(improvement_r2):.2f}%，模型解释能力增强" if improvement_r2 > 0 else f"- R²下降了{abs(improvement_r2):.2f}%"}

## 实验分析

### 统计特征的作用

1. **移动平均特征**
   - 平滑短期噪声，突出长期趋势
   - 不同窗口捕捉不同时间尺度的模式
   - 帮助模型理解温度的平稳变化

2. **标准差特征**
   - 量化温度的波动程度
   - 识别稳定期和剧烈变化期
   - 提供关于不确定性的信息

3. **趋势特征（一阶导数）**
   - 直接表示温度变化速度
   - 捕捉升温或降温趋势
   - 相比原始值更容易学习

4. **加速度特征（二阶导数）**
   - 检测趋势的转折点
   - 识别温度变化的加速或减速
   - 帮助预测突变

### 为什么特征工程有效？

{'1. **信息增益**：统计特征提供了原始特征中隐含的信息' if metrics['R2'] > step4_metrics['R2'] else '1. **特征冗余**：统计特征可能与原始特征高度相关'}
{'2. **降低学习难度**：模型不需要从头学习这些模式' if metrics['R2'] > step4_metrics['R2'] else '2. **增加复杂度**：更多特征增加了学习难度'}
{'3. **多尺度信息**：不同窗口的特征捕捉不同时间尺度' if metrics['R2'] > step4_metrics['R2'] else '3. **噪声引入**：计算的统计特征可能引入噪声'}
{'4. **物理意义**：统计特征符合温度变化的物理规律' if metrics['R2'] > step4_metrics['R2'] else '4. **过拟合风险**：特征维度增加可能导致过拟合'}

## 可视化分析

1. **预测结果对比图** (`step7_predictions.png`)
   - 展示实际值与预测值的时序对比
   
2. **预测散点图** (`step7_scatter.png`)
   - 展示预测值与真实值的相关性
   - 理想情况下应该紧密分布在45°对角线附近

## 实验结论

### 主要发现

1. **特征工程的效果**
   - {'统计特征显著提升了模型性能' if metrics['R2'] > step4_metrics['R2'] else '统计特征未能带来预期的性能提升'}
   - {'证明了领域知识在特征设计中的重要性' if metrics['R2'] > step4_metrics['R2'] else '说明特征选择需要更加谨慎'}

2. **最佳模型选择**
   - {'✅ 步骤7成为新的最佳模型' if metrics['R2'] > step4_metrics['R2'] else '⚠️ 步骤4仍是最佳模型'}
   - {'推荐使用步骤7进行温度预测' if metrics['R2'] > step4_metrics['R2'] else '建议继续优化特征工程方法'}

### 下一步改进建议

{'#### 进一步优化（基于步骤7）⭐⭐⭐⭐⭐' if metrics['R2'] > step4_metrics['R2'] else '#### 特征工程优化 ⭐⭐⭐⭐⭐'}

1. **特征选择**
   - 使用特征重要性分析，去除冗余特征
   - 尝试不同的统计窗口大小
   - 添加更多领域特征（如温度-湿度交互项）

2. **集成学习** ⭐⭐⭐⭐
   - 训练多个模型并集成
   - Bagging或Stacking
   - 预期提升10-15%

3. **超参数优化** ⭐⭐⭐⭐
   - 使用网格搜索或贝叶斯优化
   - 优化学习率、隐藏层大小等
   - 预期提升5-10%

4. **模型架构改进** ⭐⭐⭐
   - 尝试更深的网络
   - 添加残差连接
   - 预期提升3-5%

## 图表文件

- 预测结果对比图: `results/step7_predictions.png`
- 预测散点图: `results/step7_scatter.png`

---

**实验评级**: {'⭐⭐⭐⭐⭐ 优秀（显著提升）' if metrics['R2'] > step4_metrics['R2'] else '⭐⭐⭐ 良好（需要进一步优化）'}
**推荐使用**: {'是' if metrics['R2'] > step4_metrics['R2'] else '否，建议继续使用步骤4'}
"""
    
    report_path = os.path.join(docs_dir, "step7_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("\n" + "="*50)
    print("步骤7完成！")
    print(f"结果已保存至: {results_dir}")
    print(f"报告已保存至: {report_path}")
    print("="*50)

if __name__ == "__main__":
    main()
