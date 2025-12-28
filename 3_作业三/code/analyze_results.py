"""
模型性能分析脚本
"""
import pandas as pd
import numpy as np

print("=" * 80)
print("时序预测模型性能完整分析")
print("=" * 80)

# 读取数据
try:
    df_basic = pd.read_csv('../results/basic_model_comparison.csv')
    print("\n基础版模型性能")
    print("-" * 80)
    print(df_basic.to_string(index=False))
except FileNotFoundError:
    print("\n基础版对比文件未找到，可能在code目录下")
    df_basic = None

df_improved = pd.read_csv('../results/improved_model_comparison.csv')
print("\n\n改进版模型性能")
print("-" * 80)
print(df_improved.to_string(index=False))

# 性能对比
print("\n\n" + "=" * 80)
print("关键性能指标分析")
print("=" * 80)

# 找出最佳模型
best_idx = df_improved['R²'].idxmax()
best_model = df_improved.loc[best_idx]

print(f"\n🏆 最佳模型: {best_model['Model']}")
print(f"   R² Score:  {best_model['R²']:.4f} (越接近1越好)")
print(f"   RMSE:      {best_model['RMSE']:.4f} (越小越好)")
print(f"   MAE:       {best_model['MAE']:.4f} (越小越好)")
print(f"   MAPE:      {best_model['MAPE (%)']:.2f}% (越小越好)")

# 模型排名
print("\n📊 模型综合排名（按R²）:")
sorted_models = df_improved.sort_values('R²', ascending=False)
for i, (idx, row) in enumerate(sorted_models.iterrows(), 1):
    print(f"   {i}. {row['Model']:15s} - R²: {row['R²']:.4f}, RMSE: {row['RMSE']:.2f}, MAE: {row['MAE']:.2f}")

# 如果有基础版数据，进行对比
if df_basic is not None:
    print("\n\n" + "=" * 80)
    print("基础版 vs 改进版 性能提升")
    print("=" * 80)
    
    # LSTM对比
    basic_lstm = df_basic[df_basic['Model'] == 'LSTM']['R²'].values
    improved_lstm = df_improved[df_improved['Model'] == 'Improved LSTM']['R²'].values
    
    if len(basic_lstm) > 0 and len(improved_lstm) > 0:
        improvement = (improved_lstm[0] / basic_lstm[0] - 1) * 100
        print(f"\n📈 LSTM 改进:")
        print(f"   基础版 R²:  {basic_lstm[0]:.4f}")
        print(f"   改进版 R²:  {improved_lstm[0]:.4f}")
        print(f"   提升幅度:   {improvement:.2f}%")
    
    # GRU对比
    basic_gru = df_basic[df_basic['Model'] == 'GRU']['R²'].values
    improved_gru = df_improved[df_improved['Model'] == 'Improved GRU']['R²'].values
    
    if len(basic_gru) > 0 and len(improved_gru) > 0:
        improvement = (improved_gru[0] / basic_gru[0] - 1) * 100
        print(f"\n📈 GRU 改进:")
        print(f"   基础版 R²:  {basic_gru[0]:.4f}")
        print(f"   改进版 R²:  {improved_gru[0]:.4f}")
        print(f"   提升幅度:   {improvement:.2f}%")

# 详细分析
print("\n\n" + "=" * 80)
print("模型特点分析")
print("=" * 80)

for idx, row in df_improved.iterrows():
    print(f"\n{row['Model']}:")
    print(f"  ✓ R² = {row['R²']:.4f} - ", end="")
    if row['R²'] > 0.85:
        print("优秀（解释了>85%的方差）")
    elif row['R²'] > 0.80:
        print("良好（解释了>80%的方差）")
    else:
        print("可接受")
    
    print(f"  ✓ RMSE = {row['RMSE']:.2f} - 平均预测误差", end="")
    if row['RMSE'] < df_improved['RMSE'].mean():
        print(" (低于平均)")
    else:
        print(" (高于平均)")
    
    print(f"  ✓ MAE = {row['MAE']:.2f} - 绝对误差", end="")
    if row['MAE'] < df_improved['MAE'].mean():
        print(" (低于平均)")
    else:
        print(" (高于平均)")
    
    print(f"  ✓ MAPE = {row['MAPE (%)']:.2f}% - 相对误差百分比")

# 优化建议
print("\n\n" + "=" * 80)
print("💡 优化空间分析")
print("=" * 80)

# 检查性能差异
r2_range = df_improved['R²'].max() - df_improved['R²'].min()
rmse_range = df_improved['RMSE'].max() - df_improved['RMSE'].min()

print(f"\n当前模型间性能差异:")
print(f"  • R² 差异: {r2_range:.4f} ({r2_range/df_improved['R²'].mean()*100:.2f}%)")
print(f"  • RMSE 差异: {rmse_range:.4f} ({rmse_range/df_improved['RMSE'].mean()*100:.2f}%)")

print("\n🔍 可能的优化方向:")

# 1. 模型架构优化
if r2_range < 0.01:
    print("\n1. 模型架构优化潜力有限")
    print("   ✓ 三个模型性能非常接近，说明当前架构已经比较优秀")
    print("   ✓ 建议: 保持当前架构，重点在超参数和特征工程")
else:
    print("\n1. 模型架构还有优化空间")
    print("   • 尝试增加隐藏层大小 (当前128 → 256)")
    print("   • 尝试增加层数 (当前2层 → 3-4层)")
    print("   • 尝试注意力机制 (Attention)")

# 2. 超参数优化
print("\n2. 超参数优化建议")
print("   • 学习率调整: 尝试学习率搜索 (0.0001 - 0.001)")
print("   • Batch size: 测试不同大小 (32, 64, 128)")
print("   • Dropout: 当前0.3，可尝试 0.2-0.4")
print("   • Window size: 当前12步，可尝试 6, 18, 24")

# 3. 特征工程
print("\n3. 特征工程优化")
print("   • 当前47个特征，可能存在冗余")
print("   • 建议: 特征重要性分析，移除不重要特征")
print("   • 尝试: 添加更长窗口的统计特征 (24h, 48h)")
print("   • 尝试: 天气状态聚类特征")

# 4. 训练策略
print("\n4. 训练策略优化")
print("   • 增加训练轮数 (当前60 → 100)")
print("   • 使用更激进的学习率调度")
print("   • 尝试 k-fold 交叉验证")
print("   • 考虑集成学习 (Ensemble)")

# 5. 数据增强
print("\n5. 数据处理优化")
print("   • 检查异常值并处理")
print("   • 尝试不同的归一化方法 (MinMax vs Standard)")
print("   • 考虑季节性分解")

# 评估当前性能
best_r2 = df_improved['R²'].max()
print(f"\n📌 当前最佳性能评估:")
if best_r2 > 0.90:
    print(f"   ✅ R² = {best_r2:.4f} - 优秀！模型表现很好")
    print("   建议: 重点在模型部署和监控，优化空间有限")
elif best_r2 > 0.85:
    print(f"   ✅ R² = {best_r2:.4f} - 良好！有一定优化空间")
    print("   建议: 可尝试上述方向1-3，预期提升3-5%")
elif best_r2 > 0.80:
    print(f"   ⚠️ R² = {best_r2:.4f} - 可接受，仍有较大优化空间")
    print("   建议: 全面尝试上述优化方向，预期提升5-10%")
else:
    print(f"   ⚠️ R² = {best_r2:.4f} - 需要改进")
    print("   建议: 重新审视问题定义和数据质量")

# MAPE分析
best_mape = df_improved.loc[best_idx, 'MAPE (%)']
print(f"\n📌 误差率评估:")
if best_mape < 10:
    print(f"   ✅ MAPE = {best_mape:.2f}% - 优秀！")
elif best_mape < 20:
    print(f"   ✅ MAPE = {best_mape:.2f}% - 良好")
elif best_mape < 30:
    print(f"   ⚠️ MAPE = {best_mape:.2f}% - 可接受，建议继续优化")
else:
    print(f"   ⚠️ MAPE = {best_mape:.2f}% - 较高，需要改进")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)
