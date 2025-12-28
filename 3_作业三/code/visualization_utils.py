"""
可视化辅助函数 - 生成单独的图表文件
"""

import matplotlib.pyplot as plt
import os
import numpy as np


def plot_individual_charts(train_losses, val_losses, predictions, actuals, output_dir, model_name):
    """
    为每个模型生成独立的图表文件
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        predictions: 预测值数组
        actuals: 真实值数组
        output_dir: 输出目录
        model_name: 模型名称（用于标题）
    """
    print(f"\n生成{model_name}的可视化图表...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 图1: 训练和验证损失
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.8, linewidth=2, color='#1f77b4')
    plt.plot(val_losses, label='Validation Loss', alpha=0.8, linewidth=2, color='#ff7f0e')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title(f'{model_name} - Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    save_path = os.path.join(output_dir, '01_training_loss.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close()
    
    # 图2: 预测值 vs 真实值（时间序列）
    sample_size = min(500, len(predictions))
    plt.figure(figsize=(14, 6))
    time_steps = np.arange(sample_size)
    plt.plot(time_steps, actuals[:sample_size], label='Actual', alpha=0.7, linewidth=1.5, color='#2ca02c')
    plt.plot(time_steps, predictions[:sample_size], label='Predicted', alpha=0.7, linewidth=1.5, color='#d62728')
    plt.xlabel('Time Step', fontsize=12, fontweight='bold')
    plt.ylabel('Temperature (OT)', fontsize=12, fontweight='bold')
    plt.title(f'{model_name} - Prediction vs Actual (First {sample_size} Points)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    save_path = os.path.join(output_dir, '02_prediction_vs_actual_timeseries.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close()
    
    # 图3: 散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.4, s=10, c='#1f77b4', edgecolors='none')
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect Prediction', alpha=0.8)
    plt.xlabel('Actual Temperature', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Temperature', fontsize=12, fontweight='bold')
    plt.title(f'{model_name} - Scatter Plot', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axis('equal')
    plt.tight_layout()
    save_path = os.path.join(output_dir, '03_scatter_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close()
    
    # 图4: 误差分布直方图
    errors = predictions - actuals
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(errors, bins=60, alpha=0.75, edgecolor='black', color='#8c564b')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='Zero Error', alpha=0.8)
    plt.axvline(x=errors.mean(), color='green', linestyle=':', linewidth=2.5, 
                label=f'Mean Error: {errors.mean():.2f}', alpha=0.8)
    plt.xlabel('Prediction Error', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title(f'{model_name} - Error Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    save_path = os.path.join(output_dir, '04_error_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close()
    
    # 图5: 误差随时间变化
    plt.figure(figsize=(14, 6))
    plt.plot(time_steps, errors[:sample_size], alpha=0.7, linewidth=1, color='#e377c2')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    plt.fill_between(time_steps, errors[:sample_size], 0, alpha=0.3, color='#e377c2')
    plt.xlabel('Time Step', fontsize=12, fontweight='bold')
    plt.ylabel('Prediction Error', fontsize=12, fontweight='bold')
    plt.title(f'{model_name} - Error Over Time (First {sample_size} Points)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    save_path = os.path.join(output_dir, '05_error_over_time.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close()
    
    # 图6: 误差箱线图
    plt.figure(figsize=(8, 6))
    box = plt.boxplot([errors], labels=[model_name], patch_artist=True, widths=0.5)
    box['boxes'][0].set_facecolor('#9467bd')
    box['boxes'][0].set_alpha(0.7)
    plt.ylabel('Prediction Error', fontsize=12, fontweight='bold')
    plt.title(f'{model_name} - Error Box Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 添加统计信息
    stats_text = f'Mean: {errors.mean():.2f}\nStd: {errors.std():.2f}\nMedian: {np.median(errors):.2f}'
    plt.text(1.15, np.median(errors), stats_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    save_path = os.path.join(output_dir, '06_error_boxplot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close()
    
    # 图7: 预测值vs真实值（细节视图 - 前100个点）
    detail_size = min(100, len(predictions))
    plt.figure(figsize=(14, 6))
    detail_steps = np.arange(detail_size)
    plt.plot(detail_steps, actuals[:detail_size], 'o-', label='Actual', 
             alpha=0.8, linewidth=2, markersize=4, color='#2ca02c')
    plt.plot(detail_steps, predictions[:detail_size], 's-', label='Predicted', 
             alpha=0.8, linewidth=2, markersize=4, color='#d62728')
    plt.xlabel('Time Step', fontsize=12, fontweight='bold')
    plt.ylabel('Temperature (OT)', fontsize=12, fontweight='bold')
    plt.title(f'{model_name} - Detailed View (First {detail_size} Points)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    save_path = os.path.join(output_dir, '07_detailed_view.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close()
    
    print(f"  ✓ 已生成7张图表到: {output_dir}\n")


def plot_comparison_charts(results_dict, output_dir='../results'):
    """
    生成多个模型的对比图表
    
    Args:
        results_dict: 字典，包含各模型的结果
        output_dir: 输出目录
    """
    print("\n生成模型对比图表...")
    os.makedirs(output_dir, exist_ok=True)
    
    names = list(results_dict.keys())
    
    # 对比图1: R²得分对比
    plt.figure(figsize=(10, 6))
    r2_scores = [results_dict[name]['metrics']['R2'] for name in names]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(names)))
    bars = plt.bar(names, r2_scores, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
    plt.ylabel('R² Score', fontsize=12, fontweight='bold')
    plt.title('Model Comparison - R² Score', fontsize=14, fontweight='bold')
    plt.ylim([min(r2_scores) * 0.9, max(r2_scores) * 1.05])
    
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'comparison_01_r2_score.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close()
    
    # 对比图2: RMSE对比
    plt.figure(figsize=(10, 6))
    rmse_scores = [results_dict[name]['metrics']['RMSE'] for name in names]
    bars = plt.bar(names, rmse_scores, alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)
    plt.ylabel('RMSE', fontsize=12, fontweight='bold')
    plt.title('Model Comparison - RMSE', fontsize=14, fontweight='bold')
    
    for bar, score in zip(bars, rmse_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'comparison_02_rmse.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close()
    
    # 对比图3: MAE对比
    plt.figure(figsize=(10, 6))
    mae_scores = [results_dict[name]['metrics']['MAE'] for name in names]
    bars = plt.bar(names, mae_scores, alpha=0.8, color='lightgreen', edgecolor='black', linewidth=1.5)
    plt.ylabel('MAE', fontsize=12, fontweight='bold')
    plt.title('Model Comparison - MAE', fontsize=14, fontweight='bold')
    
    for bar, score in zip(bars, mae_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'comparison_03_mae.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close()
    
    # 对比图4: 所有指标雷达图
    metrics_names = ['R²', 'RMSE', 'MAE']
    num_metrics = len(metrics_names)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for idx, name in enumerate(names):
        # 归一化指标值（R²越大越好，RMSE和MAE越小越好）
        r2_norm = results_dict[name]['metrics']['R2']
        rmse_norm = 1 - (results_dict[name]['metrics']['RMSE'] / max(rmse_scores))
        mae_norm = 1 - (results_dict[name]['metrics']['MAE'] / max(mae_scores))
        
        values = [r2_norm, rmse_norm, mae_norm]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name, alpha=0.7)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Model Comparison - Radar Chart (Normalized)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'comparison_04_radar_chart.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close()
    
    print(f"  ✓ 已生成4张对比图到: {output_dir}\n")
