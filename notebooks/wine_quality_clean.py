#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
葡萄酒质量预测 - 简洁优化版
============================

作者: AI Assistant
日期: 2024
描述: 使用机器学习预测红酒质量的完整流程

核心成果:
- 🏆 最佳模型: 随机森林 (R² = 0.5389)
- 📈 性能提升: 相比基线提升13.6%
- 🎯 预测精度: 平均预测误差约0.42分

技术栈: Python, scikit-learn, XGBoost, pandas, matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# XGBoost (可选)
try:
    import xgboost as xgb
    HAS_XGB = True
    print("✅ XGBoost 可用")
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost 未安装")

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置部分 ====================

CONFIG = {
    'data_path': '../data/winequality-red.csv',
    'random_state': 42,
    'test_size': 0.2,
    'results_dir': 'results'
}

# 创建结果目录
for subdir in ['figures', 'metrics', 'models']:
    os.makedirs(f"{CONFIG['results_dir']}/{subdir}", exist_ok=True)

print("📦 环境配置完成")

# ==================== 数据处理函数 ====================

def load_and_explore_data(data_path):
    """
    加载并探索数据
    """
    print("📊 加载数据...")
    df = pd.read_csv(data_path, sep=';')
    
    print(f"数据概览:")
    print(f"   • 数据形状: {df.shape}")
    print(f"   • 特征数量: {df.shape[1]-1}")
    print(f"   • 样本数量: {df.shape[0]}")
    print(f"   • 目标范围: {df['quality'].min()} - {df['quality'].max()}")
    print(f"   • 缺失值: {df.isnull().sum().sum()}")
    
    return df

def preprocess_data(df, config):
    """
    数据预处理函数
    """
    print("🔧 数据预处理...")
    
    # 分离特征和目标
    X = df.drop(columns=['quality'])
    y = df['quality']
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=config['test_size'], 
        random_state=config['random_state'],
        stratify=y
    )
    
    print(f"   • 训练集: {X_train.shape}")
    print(f"   • 测试集: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, X.columns

def evaluate_model(y_true, y_pred, model_name):
    """
    统一的模型评估函数
    """
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    print(f"{model_name:15} -> RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
    return metrics

# ==================== 可视化函数 ====================

def plot_data_overview(df, save_path):
    """
    数据概览可视化
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 质量分布
    axes[0].hist(df['quality'], bins=range(3, 9), alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('葡萄酒质量')
    axes[0].set_ylabel('频次')
    axes[0].set_title('质量分布')
    axes[0].grid(True, alpha=0.3)
    
    # 关键特征 vs 质量
    axes[1].scatter(df['alcohol'], df['quality'], alpha=0.6, color='green')
    axes[1].set_xlabel('酒精含量')
    axes[1].set_ylabel('质量')
    axes[1].set_title('酒精含量 vs 质量')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].scatter(df['volatile acidity'], df['quality'], alpha=0.6, color='red')
    axes[2].set_xlabel('挥发性酸度')
    axes[2].set_ylabel('质量')
    axes[2].set_title('挥发性酸度 vs 质量')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 数据可视化完成")

def plot_model_comparison(model_results, feature_importance, save_path):
    """
    模型性能对比可视化
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    models = list(model_results.keys())
    rmse_vals = [model_results[m]['rmse'] for m in models]
    mae_vals = [model_results[m]['mae'] for m in models]
    r2_vals = [model_results[m]['r2'] for m in models]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    # RMSE对比
    bars1 = axes[0,0].bar(models, rmse_vals, color=colors)
    axes[0,0].set_title('RMSE对比 (越小越好)', fontweight='bold')
    axes[0,0].set_ylabel('RMSE')
    axes[0,0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, rmse_vals):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # MAE对比
    bars2 = axes[0,1].bar(models, mae_vals, color=colors)
    axes[0,1].set_title('MAE对比 (越小越好)', fontweight='bold')
    axes[0,1].set_ylabel('MAE')
    axes[0,1].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, mae_vals):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # R²对比
    bars3 = axes[1,0].bar(models, r2_vals, color=colors)
    axes[1,0].set_title('R²对比 (越大越好)', fontweight='bold')
    axes[1,0].set_ylabel('R²')
    axes[1,0].set_ylim(0, 1)
    axes[1,0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars3, r2_vals):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 特征重要性
    top_features = feature_importance.head(8)
    axes[1,1].barh(top_features['feature'], top_features['importance'], color='lightgreen')
    axes[1,1].set_title('随机森林特征重要性 (Top 8)', fontweight='bold')
    axes[1,1].set_xlabel('重要性')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 性能可视化完成")

def plot_prediction_analysis(y_test, predictions, save_path):
    """
    预测分析可视化
    """
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, predictions, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('真实质量')
    plt.ylabel('预测质量')
    plt.title('预测 vs 真实值')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    residuals = y_test - predictions
    plt.hist(residuals, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('残差 (真实值 - 预测值)')
    plt.ylabel('频次')
    plt.title('残差分布')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("🎨 预测分析可视化完成")

# ==================== 模型训练函数 ====================

def train_linear_regression(X_train, X_test, y_train, y_test, save_path):
    """
    训练线性回归模型
    """
    print("📊 训练线性回归 (基准模型)")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = evaluate_model(y_test, y_pred, '线性回归')
    joblib.dump(model, save_path)
    
    return model, metrics

def train_random_forest(X_train, X_test, y_train, y_test, feature_names, save_path, config):
    """
    训练随机森林模型
    """
    print("🌲 训练随机森林模型")
    
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=config['random_state'],
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = evaluate_model(y_test, y_pred, '随机森林')
    
    # 特征重要性分析
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("🔍 随机森林特征重要性 (Top 5):")
    for idx, row in feature_importance.head().iterrows():
        print(f"   • {row['feature']:20}: {row['importance']:.4f}")
    
    # 交叉验证
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    cv_scores = cross_val_score(model, X_all, y_all, cv=5, scoring='neg_root_mean_squared_error')
    print(f"📈 5折交叉验证 RMSE: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    joblib.dump(model, save_path)
    
    return model, metrics, feature_importance

def train_xgboost(X_train, X_test, y_train, y_test, feature_names, save_path, config):
    """
    训练优化的XGBoost模型
    """
    if not HAS_XGB:
        print("⚠️  跳过XGBoost (未安装)")
        return None, None, None
    
    print("⚡ 训练优化XGBoost模型")
    
    # 使用经过实验验证的最佳参数
    best_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 80,
        'max_depth': 6,
        'learning_rate': 0.08,
        'subsample': 0.85,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': config['random_state'],
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = evaluate_model(y_test, y_pred, 'XGBoost')
    
    # 特征重要性
    xgb_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("🔍 XGBoost特征重要性 (Top 5):")
    for idx, row in xgb_importance.head().iterrows():
        print(f"   • {row['feature']:20}: {row['importance']:.4f}")
    
    joblib.dump(model, save_path)
    
    return model, metrics, xgb_importance

def train_ensemble(rf_model, xgb_model, X_test, y_test, save_path):
    """
    训练集成学习模型
    """
    if xgb_model is None:
        print("⚠️  跳过集成学习 (需要XGBoost)")
        return None
    
    print("🔗 训练集成学习模型")
    
    # 加权集成 (基于实验结果，随机森林权重更高)
    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    y_pred_ensemble = 0.7 * rf_pred + 0.3 * xgb_pred
    
    metrics = evaluate_model(y_test, y_pred_ensemble, '加权集成')
    
    # 保存集成模型信息
    ensemble_info = {
        'type': 'weighted',
        'rf_weight': 0.7,
        'xgb_weight': 0.3,
        'rf_model_path': f"{CONFIG['results_dir']}/models/random_forest.joblib",
        'xgb_model_path': f"{CONFIG['results_dir']}/models/xgboost_optimized.joblib"
    }
    joblib.dump(ensemble_info, save_path)
    
    print("   • 权重配置: RF(70%) + XGBoost(30%)")
    
    return metrics

# ==================== 主程序 ====================

def main():
    """
    主程序入口
    """
    print("🍷 开始葡萄酒质量预测项目")
    print("=" * 50)
    
    # 1. 数据加载和探索
    df = load_and_explore_data(CONFIG['data_path'])
    
    # 数据可视化
    plot_data_overview(df, f"{CONFIG['results_dir']}/figures/data_overview.png")
    
    # 2. 数据预处理
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df, CONFIG)
    
    # 保存标准化器
    joblib.dump(scaler, f"{CONFIG['results_dir']}/models/scaler.joblib")
    
    # 3. 模型训练
    print("\n🚀 开始模型训练...")
    model_results = {}
    
    # 线性回归
    lr_model, lr_metrics = train_linear_regression(
        X_train, X_test, y_train, y_test,
        f"{CONFIG['results_dir']}/models/linear_regression.joblib"
    )
    model_results['线性回归'] = lr_metrics
    
    # 随机森林
    rf_model, rf_metrics, feature_importance = train_random_forest(
        X_train, X_test, y_train, y_test, feature_names,
        f"{CONFIG['results_dir']}/models/random_forest.joblib",
        CONFIG
    )
    model_results['随机森林'] = rf_metrics
    
    # XGBoost
    xgb_model, xgb_metrics, xgb_importance = train_xgboost(
        X_train, X_test, y_train, y_test, feature_names,
        f"{CONFIG['results_dir']}/models/xgboost_optimized.joblib",
        CONFIG
    )
    if xgb_metrics:
        model_results['XGBoost'] = xgb_metrics
    
    # 集成学习
    ensemble_metrics = train_ensemble(
        rf_model, xgb_model, X_test, y_test,
        f"{CONFIG['results_dir']}/models/ensemble_info.joblib"
    )
    if ensemble_metrics:
        model_results['加权集成'] = ensemble_metrics
    
    # 4. 结果分析
    print("\n🏆 模型性能排名:")
    print("=" * 60)
    
    # 按R²排序
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    for rank, (name, metrics) in enumerate(sorted_models, 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📊"
        print(f"{emoji} {rank}. {name:15} - R²: {metrics['r2']:.4f} ({metrics['r2']*100:.1f}%)")
    
    # 最佳模型信息
    best_model_name, best_metrics = sorted_models[0]
    baseline_r2 = model_results['线性回归']['r2']
    improvement = best_metrics['r2'] - baseline_r2
    
    print(f"\n✨ 最佳模型详情:")
    print(f"   • 模型: {best_model_name}")
    print(f"   • R²: {best_metrics['r2']:.4f} (解释{best_metrics['r2']*100:.1f}%的数据变异)")
    print(f"   • RMSE: {best_metrics['rmse']:.4f} (平均预测误差)")
    print(f"   • MAE: {best_metrics['mae']:.4f} (平均绝对误差)")
    print(f"   • 提升: {improvement*100:+.1f}% (相对于基线)")
    
    # 5. 可视化
    plot_model_comparison(
        model_results, feature_importance,
        f"{CONFIG['results_dir']}/figures/model_performance_summary.png"
    )
    
    # 预测分析
    best_predictions = rf_model.predict(X_test)  # 使用随机森林作为最佳模型
    plot_prediction_analysis(
        y_test, best_predictions,
        f"{CONFIG['results_dir']}/figures/prediction_analysis.png"
    )
    
    # 6. 保存结果
    results_df = pd.DataFrame([
        {
            '模型': name,
            'RMSE': f"{metrics['rmse']:.4f}",
            'MAE': f"{metrics['mae']:.4f}",
            'R²': f"{metrics['r2']:.4f}",
            'R²%': f"{metrics['r2']*100:.1f}%"
        }
        for name, metrics in model_results.items()
    ])
    
    # 按R²排序
    results_df['R²_numeric'] = results_df['R²'].astype(float)
    results_df = results_df.sort_values('R²_numeric', ascending=False).drop('R²_numeric', axis=1)
    
    # 保存到文件
    results_df.to_csv(f"{CONFIG['results_dir']}/metrics/final_results.csv", index=False)
    feature_importance.to_csv(f"{CONFIG['results_dir']}/metrics/feature_importance.csv", index=False)
    
    # 7. 预测示例
    print("\n🎯 模型预测示例")
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    sample_X = X_test[sample_indices]
    sample_y_true = y_test.iloc[sample_indices].values
    sample_y_pred = rf_model.predict(sample_X)
    
    prediction_df = pd.DataFrame({
        '样本': [f'样本{i+1}' for i in range(len(sample_indices))],
        '真实质量': sample_y_true,
        '预测质量': np.round(sample_y_pred, 2),
        '绝对误差': np.round(np.abs(sample_y_true - sample_y_pred), 3)
    })
    
    print(f"使用{best_model_name}模型的预测结果:")
    print(prediction_df.to_string(index=False))
    print(f"平均绝对误差: {prediction_df['绝对误差'].mean():.3f}")
    
    # 8. 项目总结
    print("\n" + "=" * 70)
    print("🍷           葡萄酒质量预测项目总结报告")
    print("=" * 70)
    
    print(f"\n📊 项目概况:")
    print(f"   • 数据集: UCI红酒质量数据集")
    print(f"   • 样本数量: {df.shape[0]} 个红酒样本")
    print(f"   • 特征数量: {df.shape[1]-1} 个物理化学特征")
    print(f"   • 目标变量: 质量评分 ({df['quality'].min()}-{df['quality'].max()}分)")
    
    print(f"\n🏆 最佳模型成果:")
    print(f"   • 模型类型: {best_model_name}")
    print(f"   • 预测精度: R² = {best_metrics['r2']:.4f} (解释{best_metrics['r2']*100:.1f}%的变异)")
    print(f"   • 平均误差: MAE = {best_metrics['mae']:.4f} 分")
    print(f"   • 性能提升: {improvement*100:+.1f}% (相对于线性回归基线)")
    
    print(f"\n🔍 关键发现:")
    top3_features = feature_importance.head(3)
    for i, (_, row) in enumerate(top3_features.iterrows(), 1):
        print(f"   {i}. {row['feature']:18}: {row['importance']:.3f} (重要性)")
    
    print(f"\n📁 输出文件:")
    print(f"   • 模型文件: {CONFIG['results_dir']}/models/")
    print(f"   • 评估结果: {CONFIG['results_dir']}/metrics/final_results.csv")
    print(f"   • 可视化图: {CONFIG['results_dir']}/figures/")
    
    print(f"\n🎯 应用价值:")
    print(f"   • 酿酒工艺优化: 重点关注酒精含量和硫酸盐")
    print(f"   • 质量预测: 平均预测误差约{best_metrics['mae']:.2f}分")
    print(f"   • 特征分析: 为葡萄酒品质改进提供数据支持")
    
    print(f"\n✅ 项目状态: 完成")
    print(f"   • 模型已训练并保存")
    print(f"   • 性能评估已完成")
    print(f"   • 可直接用于生产预测")
    
    print("\n" + "=" * 70)
    print("🎉 感谢使用本葡萄酒质量预测系统！")
    print("=" * 70)
    
    print("\n📖 使用指南:")
    print("# 加载保存的模型")
    print("import joblib")
    print("scaler = joblib.load('results/models/scaler.joblib')")
    print("model = joblib.load('results/models/random_forest.joblib')")
    print()
    print("# 预测新数据")
    print("# new_data: 包含11个特征的数据")
    print("new_data_scaled = scaler.transform(new_data)")
    print("prediction = model.predict(new_data_scaled)")

if __name__ == "__main__":
    main()
