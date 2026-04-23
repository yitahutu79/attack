
#!/usr/bin/env python3
"""TCN-GAN模型创新点可视化脚本。"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录。论文和汇报图片统一放在 attack/paper/figures，避免散落到多个目录。
output_dir = "attack/paper/figures"
os.makedirs(output_dir, exist_ok=True)

def plot_model_architecture():
    """绘制TCN-GAN模型架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # 绘制TCN生成器
    ax.text(0.1, 0.9, "TCN生成器", fontsize=14, weight='bold')
    ax.text(0.1, 0.8, "潜在空间 → 初始映射 → TCN网络 → 输出", fontsize=12)

    # 绘制TCN判别器
    ax.text(0.5, 0.9, "TCN判别器", fontsize=14, weight='bold')
    ax.text(0.5, 0.8, "输入序列 → TCN特征提取 → 注意力机制 → 异常分数", fontsize=12)

    # 绘制创新点
    ax.text(0.1, 0.6, "TCN创新点:", fontsize=12, weight='bold')
    ax.text(0.1, 0.55, "• 残差连接", fontsize=10)
    ax.text(0.1, 0.5, "• 多尺度扩张卷积", fontsize=10)
    ax.text(0.1, 0.45, "• 因果卷积", fontsize=10)

    ax.text(0.5, 0.6, "GAN创新点:", fontsize=12, weight='bold')
    ax.text(0.5, 0.55, "• WGAN-GP损失函数", fontsize=10)
    ax.text(0.5, 0.5, "• Critic与Generator分离训练", fontsize=10)
    ax.text(0.5, 0.45, "• 梯度惩罚技术", fontsize=10)

    ax.text(0.1, 0.35, "评分机制创新:", fontsize=12, weight='bold')
    ax.text(0.1, 0.3, "• 概率评分 (1-D(x))", fontsize=10)
    ax.text(0.1, 0.25, "• 特征偏差评分 (L2/马氏距离)", fontsize=10)
    ax.text(0.1, 0.2, "• 融合评分 (α·prob + (1-α)·norm(feat_dev))", fontsize=10)

    ax.text(0.5, 0.35, "可解释性(XAI):", fontsize=12, weight='bold')
    ax.text(0.5, 0.3, "• 基于梯度输入的归因分析", fontsize=10)
    ax.text(0.5, 0.25, "• 时间注意力权重可视化", fontsize=10)
    ax.text(0.5, 0.2, "• 关键特征和时间步识别", fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    axis = plt.gca()
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tcn_gan_architecture.png"), dpi=150)
    plt.close()

def plot_temporal_block():
    """绘制TemporalBlock结构图"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 绘制输入
    ax.text(0.05, 0.5, "输入\nx", fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    # 绘制第一个卷积块
    ax.text(0.25, 0.7, "因果卷积1", fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax.text(0.25, 0.3, "ReLU + Dropout", fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

    # 绘制第二个卷积块
    ax.text(0.45, 0.7, "因果卷积2", fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax.text(0.45, 0.3, "ReLU + Dropout", fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

    # 绘制残差连接
    ax.text(0.65, 0.5, "残差连接\n(可选)", fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    # 绘制输出
    ax.text(0.85, 0.5, "输出\nF(x)", fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))

    # 绘制连接线
    ax.arrow(0.15, 0.5, 0.07, 0.2, head_width=0.02, head_length=0.03, fc='black', ec='black')
    ax.arrow(0.15, 0.5, 0.07, -0.2, head_width=0.02, head_length=0.03, fc='black', ec='black')
    ax.arrow(0.35, 0.7, 0.07, 0, head_width=0.02, head_length=0.03, fc='black', ec='black')
    ax.arrow(0.35, 0.3, 0.07, 0, head_width=0.02, head_length=0.03, fc='black', ec='black')
    ax.arrow(0.55, 0.7, 0.07, -0.2, head_width=0.02, head_length=0.03, fc='black', ec='black')
    ax.arrow(0.55, 0.3, 0.07, 0.2, head_width=0.02, head_length=0.03, fc='black', ec='black')
    ax.arrow(0.55, 0.5, 0.07, 0, head_width=0.02, head_length=0.03, fc='black', ec='black')
    ax.arrow(0.75, 0.5, 0.07, 0, head_width=0.02, head_length=0.03, fc='black', ec='black')

    # 添加说明
    ax.text(0.5, 0.1, "残差连接: F(x) + x (当输入输出维度相同时)", fontsize=10, ha='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    axis = plt.gca()
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "temporal_block.png"), dpi=150)
    plt.close()

def plot_score_modes():
    """绘制不同评分模式的示意图"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 绘制概率评分模式
    ax.text(0.1, 0.8, "1. 概率评分模式", fontsize=12, weight='bold')
    ax.text(0.1, 0.75, "异常分数 = 1 - D(x)", fontsize=10)

    # 绘制简单图示
    ax.plot([0.2, 0.8], [0.6, 0.6], 'k-', lw=2)
    ax.plot([0.2, 0.8], [0.5, 0.5], 'k--', lw=1)
    ax.text(0.5, 0.65, "正常样本", fontsize=10, ha='center')
    ax.text(0.5, 0.45, "异常样本", fontsize=10, ha='center')
    ax.arrow(0.3, 0.6, 0, -0.08, head_width=0.02, head_length=0.02, fc='red', ec='red')
    ax.text(0.35, 0.52, "高分数", fontsize=9, color='red')

    # 绘制特征偏差评分模式
    ax.text(0.1, 0.35, "2. 特征偏差评分模式", fontsize=12, weight='bold')
    ax.text(0.1, 0.3, "异常分数 = ||特征向量 - 正常分布均值||", fontsize=10)

    # 绘制特征空间示意图
    circle = plt.Circle((0.5, 0.15), 0.05, color='blue', alpha=0.3)
    ax.add_patch(circle)
    ax.text(0.5, 0.15, "正常\n分布", fontsize=9, ha='center', va='center')
    ax.plot(0.6, 0.15, 'ro', markersize=8)
    ax.text(0.65, 0.15, "异常样本", fontsize=9, color='red')
    ax.plot([0.5, 0.6], [0.15, 0.15], 'k-', lw=1)
    ax.text(0.55, 0.2, "偏差距离", fontsize=9)

    # 绘制融合评分模式
    ax.text(0.1, 0.05, "3. 融合评分模式", fontsize=12, weight='bold')
    ax.text(0.1, 0.0, "异常分数 = α·prob + (1-α)·norm(feat_dev)", fontsize=10)

    # 绘制融合示意图
    ax.plot([0.2, 0.3], [0.05, 0.05], 'b-', lw=2, label='概率评分')
    ax.plot([0.4, 0.5], [0.05, 0.05], 'r-', lw=2, label='特征偏差评分')
    ax.plot([0.6, 0.7], [0.05, 0.05], 'purple', lw=3, label='融合评分')
    ax.text(0.25, 0.08, "低", fontsize=9, ha='center')
    ax.text(0.45, 0.08, "中", fontsize=9, ha='center')
    ax.text(0.65, 0.08, "高", fontsize=9, ha='center')
    ax.text(0.25, 0.02, "正常", fontsize=9, ha='center', color='blue')
    ax.text(0.45, 0.02, "中等", fontsize=9, ha='center', color='orange')
    ax.text(0.65, 0.02, "异常", fontsize=9, ha='center', color='red')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    axis = plt.gca()
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_modes.png"), dpi=150)
    plt.close()

def plot_xai_methods():
    """绘制XAI方法示意图"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 绘制基于梯度输入的归因分析
    ax.text(0.1, 0.8, "XAI方法: 基于梯度输入的归因分析", fontsize=12, weight='bold')
    ax.text(0.1, 0.75, "归因值 = |∂异常分数/∂输入| × |输入|", fontsize=10)

    # 绘制时间维度重要性
    ax.text(0.1, 0.65, "时间维度重要性", fontsize=11, weight='bold')
    time_points = np.arange(10)
    normal_importance = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    anomaly_importance = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.6, 0.5, 0.4, 0.3])

    ax.plot(time_points, normal_importance, 'b-o', label='正常样本')
    ax.plot(time_points, anomaly_importance, 'r-o', label='异常样本')
    ax.set_xlabel('时间步', fontsize=10)
    ax.set_ylabel('重要性', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 绘制特征维度重要性
    ax.text(0.55, 0.65, "特征维度重要性", fontsize=11, weight='bold')
    features = ['F1', 'F2', 'F3', 'F4', 'F5']
    importance = np.array([0.2, 0.8, 0.5, 0.3, 0.7])
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']

    bars = ax.bar(features, importance, color=colors)
    ax.set_ylabel('重要性', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加说明
    ax.text(0.1, 0.2, "关键发现:", fontsize=11, weight='bold')
    ax.text(0.1, 0.15, "• 异常样本在特定时间步有更高的重要性", fontsize=10)
    ax.text(0.1, 0.1, "• 特定特征（如F2、F5）对异常检测贡献最大", fontsize=10)
    ax.text(0.1, 0.05, "• 可解释性帮助理解模型决策依据", fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    axis = plt.gca()
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "xai_methods.png"), dpi=150)
    plt.close()

def plot_training_strategy():
    """绘制训练策略示意图"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 绘制训练流程
    ax.text(0.1, 0.9, "TCN-GAN训练策略", fontsize=14, weight='bold')

    # 第一阶段：GAN训练
    ax.text(0.1, 0.8, "第一阶段: GAN训练", fontsize=12, weight='bold')
    ax.text(0.1, 0.75, "• 仅使用正常样本训练", fontsize=10)
    ax.text(0.1, 0.7, "• 生成器学习生成正常流量", fontsize=10)
    ax.text(0.1, 0.65, "• 判别器学习区分真实/生成样本", fontsize=10)

    # 绘制GAN训练流程
    ax.plot([0.1, 0.3], [0.6, 0.6], 'k-', lw=2)
    ax.plot([0.3, 0.3], [0.55, 0.65], 'k-', lw=2)
    ax.plot([0.3, 0.5], [0.6, 0.6], 'k-', lw=2)
    ax.text(0.2, 0.62, "噪声", fontsize=9, ha='center')
    ax.text(0.4, 0.62, "生成样本", fontsize=9, ha='center')
    ax.text(0.2, 0.52, "真实样本", fontsize=9, ha='center')
    ax.text(0.4, 0.52, "判别器", fontsize=9, ha='center')

    # 第二阶段：异常检测
    ax.text(0.6, 0.8, "第二阶段: 异常检测", fontsize=12, weight='bold')
    ax.text(0.6, 0.75, "• 使用训练好的判别器", fontsize=10)
    ax.text(0.6, 0.7, "• 计算异常分数", fontsize=10)
    ax.text(0.6, 0.65, "• 标定阈值控制误报率", fontsize=10)

    # 绘制异常检测流程
    ax.plot([0.6, 0.8], [0.6, 0.6], 'k-', lw=2)
    ax.plot([0.8, 0.8], [0.55, 0.65], 'k-', lw=2)
    ax.plot([0.8, 1.0], [0.6, 0.6], 'k-', lw=2)
    ax.text(0.7, 0.62, "输入序列", fontsize=9, ha='center')
    ax.text(0.9, 0.62, "异常分数", fontsize=9, ha='center')
    ax.text(0.9, 0.52, "阈值", fontsize=9, ha='center')

    # 添加创新点
    ax.text(0.1, 0.4, "训练创新点:", fontsize=12, weight='bold')
    ax.text(0.1, 0.35, "• WGAN-GP损失函数提高训练稳定性", fontsize=10)
    ax.text(0.1, 0.3, "• Critic与Generator分离训练策略", fontsize=10)
    ax.text(0.1, 0.25, "• 梯度惩罚技术避免模式崩溃", fontsize=10)

    ax.text(0.6, 0.4, "评估创新点:", fontsize=12, weight='bold')
    ax.text(0.6, 0.35, "• 多种评分机制提高检测鲁棒性", fontsize=10)
    ax.text(0.6, 0.3, "• 阈值标定机制满足部署需求", fontsize=10)
    ax.text(0.6, 0.25, "• 多种评估指标全面评估性能", fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    axis = plt.gca()
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_strategy.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    print("正在生成TCN-GAN模型创新点可视化图表...")
    plot_model_architecture()
    plot_temporal_block()
    plot_score_modes()
    plot_xai_methods()
    plot_training_strategy()
    print(f"图表已保存到 {output_dir} 目录")
