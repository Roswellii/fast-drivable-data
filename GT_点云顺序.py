import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 定义文件路径
pointcloud_path = "F:\\Workspace\\fast-drivable\\GT\\semanticKITTI\\pc\\000000.bin"  # 替换为点云文件所在路径
label_path = "F:\\Workspace\\fast-drivable\\GT\\semanticKITTI\\label\\000000.label"  # 替换为标签文件所在路径

def load_pointcloud(filepath):
    """加载点云数据 (.bin 文件)"""
    return np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)

def load_labels(filepath):
    """加载标签数据 (.label 文件)"""
    return np.fromfile(filepath, dtype=np.uint32)

def visualize_2d(points, labels):
    """在2D平面上可视化点云和标签，并显示图例"""
    # 只取前10000个点
    num_points = 4000
    points = points[:num_points]
    labels = labels[:num_points]

    # 每1000个点画一张图
    points_per_plot = 1000
    num_plots = num_points // points_per_plot

    fig, axes = plt.subplots(1, num_plots, figsize=(20, 10))

    # 确定坐标轴范围
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

    # 按每1000个点绘制一张图
    for i in range(num_plots):
        start_idx = i * points_per_plot
        end_idx = (i + 1) * points_per_plot

        x = points[start_idx:end_idx, 0]
        y = points[start_idx:end_idx, 1]

        # 按顺序生成从红到蓝的颜色
        colors = np.linspace(0, 1, points_per_plot)  # 从0到1的线性变化

        # 创建颜色映射，从红到蓝
        cmap = plt.cm.get_cmap("coolwarm")  # coolwarm 映射从红到蓝

        ax = axes[i]
        scatter = ax.scatter(x, y, c=colors, cmap=cmap, s=10)

        ax.set_title(f"Points {start_idx} to {end_idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")

        # 标记原点
        ax.scatter(0, 0, color='red', s=50, label="Origin", zorder=5)
        ax.legend()

        # 设置坐标轴范围一致
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # 添加颜色条
        fig.colorbar(scatter, ax=ax, label="Point Index (Red to Blue)")

    plt.tight_layout()
    plt.show()

def main():
    # 加载点云
    points = load_pointcloud(pointcloud_path)
    labels = load_labels(label_path)

    # 确保点云和标签数量匹配
    assert points.shape[0] == labels.shape[0], "Point cloud and labels size mismatch!"

    # 可视化
    visualize_2d(points, labels)

if __name__ == "__main__":
    main()
