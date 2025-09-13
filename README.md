# 葡萄酒质量预测项目

## 📋 项目概述

本项目使用机器学习方法预测红酒质量，基于UCI葡萄酒数据集开发。通过多种算法对比和优化，实现了准确的质量预测模型。

### 🏆 核心成果
- **最佳模型**: 随机森林 (R² = 0.5389)
- **性能提升**: 相比基线提升13.6% 
- **预测精度**: 平均预测误差约0.42分
- **关键特征**: 酒精含量、硫酸盐、挥发性酸度

## 📁 文件结构

```
notebooks/
├── README.md                          # 项目说明文档
├── wine_quality_main.ipynb           # 完整实验过程 (1642行)
├── wine_quality_clean.py             # 整洁版Python脚本 ⭐
├── total_process.ipynb               # 基础流程版本 (219行)
├── .ipynb_checkpoints/               # Jupyter自动备份
│   ├── wine_quality_main-checkpoint.ipynb
│   └── total_process-checkpoint.ipynb
└── results/                          # 所有输出结果
    ├── models/                       # 训练好的模型
    │   ├── scaler.joblib            # 数据标准化器
    │   ├── linear_regression.joblib  # 线性回归模型
    │   ├── random_forest.joblib     # 随机森林模型 (最佳)
    │   ├── xgboost_optimized.joblib # 优化XGBoost模型
    │   └── ensemble_info.joblib     # 集成模型信息
    ├── metrics/                     # 评估结果
    │   ├── final_results.csv        # 最终模型对比
    │   ├── model_comparison.csv     # 详细对比数据
    │   └── feature_importance.csv   # 特征重要性
    └── figures/                     # 可视化图表
        ├── data_overview.png        # 数据概览
        ├── model_performance_summary.png # 模型性能对比
        └── prediction_analysis.png  # 预测分析
```

## 🚀 快速开始

### 推荐使用方式

#### 1. **生产使用** - `wine_quality_clean.py` ⭐
```bash
cd notebooks
python wine_quality_clean.py
```
- **优点**: 代码整洁、注释完整、模块化设计
- **适用**: 生产环境、代码学习、项目演示

#### 2. **完整实验** - `wine_quality_main.ipynb`
```bash
jupyter notebook wine_quality_main.ipynb
```
- **优点**: 包含所有实验过程和优化步骤
- **适用**: 研究分析、算法对比、调参记录

#### 3. **快速演示** - `total_process.ipynb`
```bash
jupyter notebook total_process.ipynb
```
- **优点**: 简洁明了、快速运行
- **适用**: 教学演示、基础学习

### 模型加载和使用

```python
import joblib
import numpy as np

# 加载已训练的模型
scaler = joblib.load('results/models/scaler.joblib')
model = joblib.load('results/models/random_forest.joblib')

# 预测新数据
# new_data: shape (n_samples, 11) 包含11个特征
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
```

## 📊 数据说明

### 数据集信息
- **来源**: UCI Machine Learning Repository
- **样本数**: 1599个红酒样本
- **特征数**: 11个物理化学特征
- **目标**: 质量评分 (3-8分)

### 关键特征
1. **alcohol** - 酒精含量 (最重要特征)
2. **sulphates** - 硫酸盐含量
3. **volatile acidity** - 挥发性酸度
4. **total sulfur dioxide** - 总二氧化硫
5. **其他** - pH值、氯化物、柠檬酸等

## 🎯 模型性能

| 模型 | RMSE | MAE | R² | 评价 |
|------|------|-----|----|----|
| **随机森林** | **0.5490** | **0.4220** | **0.5389** | **🥇 最佳** |
| 加权集成 | 0.5508 | 0.4283 | 0.5358 | 🥈 很好 |
| XGBoost | 0.5632 | 0.4367 | 0.5147 | 🥉 良好 |
| 线性回归 | 0.6245 | 0.5035 | 0.4032 | 📊 基线 |

### 评估指标说明
- **RMSE**: 均方根误差，衡量预测值与真实值的平均偏差
- **MAE**: 平均绝对误差，对异常值不敏感
- **R²**: 决定系数，表示模型解释数据变异的能力 (0-1，越大越好)

## 🔬 技术栈

### 核心库
- **Python 3.8+**
- **scikit-learn** - 机器学习算法
- **pandas** - 数据处理
- **numpy** - 数值计算
- **matplotlib** - 数据可视化

### 可选库
- **XGBoost** - 梯度提升算法 (需要单独安装)

### 安装依赖
```bash
pip install scikit-learn pandas numpy matplotlib
pip install xgboost  # 可选
```

## 📈 使用建议

### 酿酒工艺优化
基于特征重要性分析：
1. **重点关注酒精含量** (重要性: 27.1%)
2. **优化硫酸盐配比** (重要性: 14.8%)
3. **控制挥发性酸度** (重要性: 11.2%)

### 质量预测应用
- **预测精度**: 平均误差约0.42分
- **适用范围**: 3-8分红酒质量评分
- **置信度**: 解释53.9%的质量变异

## 🛠️ 开发说明

### 项目历史
1. **初始版本**: `total_process.ipynb` - 基础流程
2. **完整实验**: `wine_quality_main.ipynb` - 多轮优化
3. **生产版本**: `wine_quality_clean.py` - 整洁代码

### 代码特点
- **模块化设计**: 函数式编程，易于维护
- **完整注释**: 中文注释，便于理解
- **错误处理**: 兼容性检查，优雅降级
- **结果保存**: 自动保存模型和可视化

### 扩展建议
1. **特征工程**: 添加特征交互项、多项式特征
2. **深度学习**: 尝试神经网络模型
3. **时间序列**: 如有时间数据，可建立时序模型
4. **集成优化**: 尝试Stacking、Blending等高级集成方法

## 📞 联系信息

如有问题或建议，请通过以下方式联系：
- **项目**: 葡萄酒质量预测
- **版本**: v1.0
- **更新**: 2025年

---

*本项目展示了完整的机器学习流程，从数据探索到模型部署，适合学习和生产使用。*
