import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from tqdm import tqdm  # 导入 tqdm 库
from mpl_toolkits.mplot3d import Axes3D  # 导入 3D 绘图工具

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


def find_boundary_points(points, labels, distance_threshold=0.05):
    """查找平面上的 road 类别边界点，并显示进度条"""
    boundary_points = []

    # 使用 tqdm 包装循环来显示进度条
    for i in tqdm(range(len(points)), desc="Processing Points", unit="point"):
        if labels[i] == 9:  # 只考虑 'road' 类别
            # 当前点的坐标（只考虑 x, y）
            current_point = points[i, :2]  # 只考虑 x 和 y 坐标
            # 计算与当前点距离小于 threshold 的点
            distances = np.linalg.norm(points[:, :2] - current_point, axis=1)  # 计算 x, y 距离
            neighbors = points[distances < distance_threshold]
            neighbor_labels = labels[distances < distance_threshold]
            # 如果邻居点有不同的标签，则当前点为边界点
            if not np.all(neighbor_labels == 9):  # 如果有不同标签的邻居
                boundary_points.append(points[i])  # 保留 3D 坐标

    return np.array(boundary_points)


def visualize_3d(points, labels, boundary_points, distance_threshold=20.0):
    """在3D平面上可视化点云和标签，并显示边界线"""
    # 计算每个点到原点的距离（假设原点是点云的参考位置，通常是车辆的位置）
    distances = np.linalg.norm(points[:, :3], axis=1)  # 只考虑 X, Y, Z 坐标

    # 筛选出距离小于等于 distance_threshold 的点
    mask = distances <= distance_threshold
    filtered_points = points[mask]
    filtered_labels = labels[mask]

    # 获取 x, y, z 坐标
    x = filtered_points[:, 0]
    y = filtered_points[:, 1]
    z = filtered_points[:, 2]

    # 确保标签是整数类型
    filtered_labels = filtered_labels.flatten().astype(int)

    # 创建一个 3D 图形
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')

    # ----------------- 按标签显示 -----------------
    scatter = ax.scatter(x, y, z, c=filtered_labels, cmap="tab20", s=1)
    ax.set_title("Pointcloud Colored by Labels")

    # 显示每个标签的点数，并生成图例
    unique_labels = np.unique(filtered_labels)
    label_counts = dict(Counter(filtered_labels))

    legend_patches = []
    for label in unique_labels:
        if label in LABEL_MAP:
            label_name = LABEL_MAP[label]
            label_color = plt.cm.tab20(label % 20)  # 获取标签对应的颜色
            legend_patches.append(
                mpatches.Patch(color=label_color, label=f"{label_name}: {label_counts[label]} points"))

    # 添加图例
    ax.legend(handles=legend_patches, loc="upper right", fontsize="small")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # ----------------- 显示边界点 -----------------
    if len(boundary_points) > 0:
        boundary_x = boundary_points[:, 0]
        boundary_y = boundary_points[:, 1]
        boundary_z = boundary_points[:, 2]
        ax.scatter(boundary_x, boundary_y, boundary_z, color='red', s=10, label='Boundary Points')  # 红色标记边界点

    # 显示图形
    plt.legend(loc="upper left")
    plt.show()


def main():
    # 加载点云
    points = load_pointcloud(pointcloud_path)
    labels = load_labels(label_path)

    # 确保点云和标签数量匹配
    assert points.shape[0] == labels.shape[0], "Point cloud and labels size mismatch!"

    # 查找边界点
    boundary_points = find_boundary_points(points, labels)

    # 3D 可视化
    visualize_3d(points, labels, boundary_points)


if __name__ == "__main__":
    main()
