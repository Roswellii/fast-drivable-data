import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

# 定义文件路径
pointcloud_path = "F:\\Workspace\\fast-drivable\\GT\\semanticKITTI\\pc\\000100.npy"  # 点云文件路径 (.npy 格式)
label_path = "F:\\Workspace\\fast-drivable\\GT\\semanticKITTI\\label\\000100.npy"  # 标签文件路径 (.npy 格式)

# SemanticKITTI 标签到类别的映射
LABEL_MAP = {
    0: "unlabeled",
    1: "car",
    2: "bicycle",
    3: "motorcycle",
    4: "truck",
    5: "other-vehicle",
    6: "person",
    7: "bicyclist",
    8: "motorcyclist",
    9: "road",
    10: "parking",
    11: "sidewalk",
    12: "other-ground",
    13: "building",
    14: "fence",
    15: "vegetation",
    16: "trunk",
    17: "terrain",
    18: "pole",
    19: "traffic-sign"
}

def load_pointcloud(filepath):
    """加载点云数据 (.npy 文件)"""
    return np.load(filepath)  # 使用 np.load 读取 .npy 文件

def load_labels(filepath):
    """加载标签数据 (.npy 文件)"""
    return np.load(filepath)  # 使用 np.load 读取 .npy 文件

def visualize_2d(points, labels, distance_threshold=20.0):
    """在2D平面上可视化点云和标签，只关注20米以内的点"""
    # 计算每个点到原点的距离（假设原点是点云的参考位置，通常是车辆的位置）
    distances = np.linalg.norm(points[:, :3], axis=1)  # 只考虑 X, Y, Z 坐标

    # 筛选出距离小于等于 distance_threshold 的点
    mask = distances <= distance_threshold
    filtered_points = points[mask]
    filtered_labels = labels[mask]

    x = filtered_points[:, 0]
    y = filtered_points[:, 1]

    # 确保标签是整数类型
    filtered_labels = filtered_labels.flatten().astype(int)

    # 获取唯一的标签
    unique_labels = np.unique(filtered_labels)

    # 统计每个类别的点数
    label_counts = dict(Counter(filtered_labels))

    # 创建一个 1x2 的子图
    fig, ax = plt.subplots(1, 2, figsize=(40, 20))

    # ----------------- 第一张图：按标签显示 -----------------
    scatter = ax[0].scatter(x, y, c=filtered_labels, cmap="tab20", s=1)
    ax[0].set_title("Pointcloud Colored by Labels")

    # 显示每个标签的点数，并生成图例
    legend_patches = []
    for label in unique_labels:
        if label in LABEL_MAP:
            label_name = LABEL_MAP[label]
            label_color = plt.cm.tab20(label % 20)  # 获取标签对应的颜色
            legend_patches.append(mpatches.Patch(color=label_color, label=f"{label_name}: {label_counts[label]} points"))

    # 添加图例
    ax[0].legend(handles=legend_patches, loc="upper right", fontsize="small")
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    plt.colorbar(scatter, ax=ax[0], label="Label Index")

    # ----------------- 第二张图：按点云顺序显示渐变色 -----------------
    # 使用点云的顺序作为颜色映射，从红色到绿色
    point_order = np.arange(filtered_points.shape[0])  # 点云的顺序
    scatter = ax[1].scatter(x, y, c=point_order, cmap="RdYlGn", s=1)  # 使用 "RdYlGn" 渐变色
    ax[1].set_title("Pointcloud Colored by Point Order (Red to Green)")

    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    plt.colorbar(scatter, ax=ax[1], label="Point Order")

    # 显示图形
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
