import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    x = points[:, 0]
    y = points[:, 1]

    # 按顺序生成从红到蓝的颜色
    num_points = len(points)
    colors = np.linspace(0, 1, num_points)  # 从0到1的线性变化

    # 创建颜色映射，从红到蓝
    cmap = plt.cm.get_cmap("coolwarm")  # coolwarm 映射从红到蓝

    plt.figure(figsize=(20, 20))

    # 根据颜色值绘制点
    plt.scatter(x, y, c=colors, cmap=cmap, s=1)

    plt.axis("equal")
    plt.title("2D Visualization of Point Cloud with Labels")
    plt.xlabel("X")
    plt.ylabel("Y")

    # 添加颜色条
    plt.colorbar(label="Point Index (Red to Blue)")
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
