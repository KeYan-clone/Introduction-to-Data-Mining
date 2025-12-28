"""
快速测试脚本 - 验证目录结构和图表生成
"""

import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

from visualization_utils import plot_individual_charts
import numpy as np

print("=" * 60)
print("测试目录结构和图表生成功能")
print("=" * 60)

# 生成模拟数据
np.random.seed(42)
n_epochs = 50
n_samples = 1000

train_losses = np.exp(-np.linspace(0, 3, n_epochs)) + np.random.normal(0, 0.01, n_epochs)
val_losses = np.exp(-np.linspace(0, 2.5, n_epochs)) + np.random.normal(0, 0.02, n_epochs)

actuals = np.sin(np.linspace(0, 10, n_samples)) * 100 + 400 + np.random.normal(0, 5, n_samples)
predictions = actuals + np.random.normal(0, 10, n_samples)

# 测试各个模型目录
models = [
    ('Basic LSTM', '../results/basic_lstm'),
    ('Basic GRU', '../results/basic_gru'),
    ('Improved LSTM', '../results/improved_lstm'),
    ('Bi-LSTM', '../results/bi_lstm'),
    ('Improved GRU', '../results/improved_gru')
]

print("\n测试图表生成功能...\n")

for model_name, output_dir in models:
    print(f"生成 {model_name} 的图表...")
    plot_individual_charts(
        train_losses=train_losses,
        val_losses=val_losses,
        predictions=predictions,
        actuals=actuals,
        output_dir=output_dir,
        model_name=model_name
    )

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
print("\n请检查以下目录：")
for _, output_dir in models:
    print(f"  - {output_dir}")
print("\n每个目录应包含7张图表文件。")
