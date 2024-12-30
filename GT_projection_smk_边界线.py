import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os

# 定义文件路径
pointcloud_path = "F:\\Workspace\\fast-drivable\\GT\\semanticKITTI\\pc\\08_000300.npy"  # 点云文件路径 (.npy 格式)
# label_path = "F:\Workspace\\fast-drivable\GT\model_predictions\\08\\000000.npy"  # 标签文件路径 (.npy 格式)
label_path = "F:\\Workspace\\fast-drivable\\GT\\semanticKITTI\\label\\08_000300.npy"  # 标签文件路径 (.npy 格式)

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
    9: "road",  # road 类别
    10: "parking",
    11: "sidewalk",  # sidewalk 类别
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


def main():
    points = load_pointcloud(pointcloud_path)
    labels = load_labels(label_path)

    # 确保点云和标签数量匹配
    assert points.shape[0] == labels.shape[0], "Point cloud and labels size mismatch!"

    # 直接在主程序中进行 labels.flatten()
    labels = labels.flatten()  # 确保 labels 是一维数组

    # 过滤点云，只考虑 X 和 Y 坐标的绝对值均小于等于 20 的点
    x_mask = np.abs(points[:, 0]) <= 20.0  # X 绝对值在 20 以内
    y_mask = np.abs(points[:, 1]) <= 20.0  # Y 绝对值在 20 以内
    mask = x_mask & y_mask  # 同时满足 X 和 Y 条件

    filtered_points = points[mask]
    filtered_labels = labels[mask]

    # 查找 road 和 non-road 点
    road_points = filtered_points[filtered_labels == 9]  # 选取 'road' 类别的点
    non_road_points = filtered_points[filtered_labels != 9]  # 选取非 'road' 类别的点

    p1 = road_points
    p2 = non_road_points
    points = np.vstack((p1, p2))

    # 计算点云之间的最近邻
    nbrs1 = NearestNeighbors(n_neighbors=1).fit(p1)
    distances1, indices1 = nbrs1.kneighbors(p2)

    nbrs2 = NearestNeighbors(n_neighbors=1).fit(p2)
    distances2, indices2 = nbrs2.kneighbors(p1)

    # 找到最小距离的点对
    boundary_points_p1 = p1[indices1.flatten()]
    boundary_points_p2 = p2[indices2.flatten()]

    # 设置图形尺寸
    plt.figure(figsize=(20, 20))

    # 可视化交线
    plt.scatter(p1[:, 0], p1[:, 1], label='Road Points', alpha=0.6, s=2)
    plt.scatter(p2[:, 0], p2[:, 1], label='Non-Road Points', alpha=0.6, s=2)
    plt.scatter(boundary_points_p1[:, 0], boundary_points_p1[:, 1], color='r', label='Boundary Points from Road', s=3)

    # 保存 boundary_points_p1 数据
    base_filename = os.path.splitext(os.path.basename(pointcloud_path))[0]
    boundary_points_filename = "GT_"+ f"boundary_points_{base_filename}.npy"
    np.save(boundary_points_filename, boundary_points_p1)

    # 添加图例和标签
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Point Cloud Visualization with Road and Non-Road Boundary Points")

    # 保存图像
    plt.savefig("GT_"+f"{base_filename}.png")

    # 显示图形
    plt.show()


if __name__ == "__main__":
    main()
