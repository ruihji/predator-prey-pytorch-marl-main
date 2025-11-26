import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设定文件夹路径
folder_path = './'  # 当前文件夹，或修改为你的文件夹路径

# 获取所有文件
files = os.listdir(folder_path)

# 用于存储数据
all_data = {}

# 自动从文件名提取算法名和seed
for file in files:
    # 检查是否是 CSV 文件
    if file.endswith('.csv'):
        parts = file.split('_')
        algorithm = parts[0]
        seed = int(parts[-1].split('_')[-1].split('.')[0])
        # 加载 CSV 文件
        data = pd.read_csv(os.path.join(folder_path, file))

        # 提取 total_steps 和 evaluate_reward
        steps = data['episode']
        rewards = data['eval_metric']

        # 存储数据
        if algorithm not in all_data:
            all_data[algorithm] = []

        all_data[algorithm].append((steps, rewards))

# 绘制图表
plt.figure(figsize=(8, 6))

for alg, data_list in all_data.items():
    all_rewards = []
    all_steps = []

    for steps, rewards in data_list:
        all_rewards.append(rewards)
        all_steps.append(steps)

    # 计算每个算法的均值和标准差
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)

    # 绘制折线图并填充阴影区域
    plt.plot(all_steps[0], mean_rewards, label=alg, linewidth=2)
    plt.fill_between(all_steps[0], mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)

# 添加标题和坐标轴标签
plt.title('Performance Comparison of Algorithms', fontsize=14, fontweight='bold')
plt.xlabel('Total Steps', fontsize=12)
plt.ylabel('Evaluate Reward', fontsize=12)

# 设置图例
plt.legend(loc='upper left', fontsize=10)

# 设置网格
plt.grid(True, linestyle='--', alpha=0.3)

# 优化布局，确保标签不被裁剪
plt.tight_layout()

# 显示图表
plt.show()

plt.savefig('performance_comparison.png')
