import numpy as np
import matplotlib.pyplot as plt
import os

# 定义基本路径和文件夹编号列表
base_path = "F:/Workspace/fast-drivable/GT/boundary_points/"
folder_ids = ["08_000160"]  # 你可以扩展更多的文件夹编号

def load_points(filepath):
    """加载点云数据 (.npy 文件)"""
    return np.load(filepath)  # 使用 np.load 读取 .npy 文件

def process_folder(folder_id):
    # 构造文件路径
    boundary_points_1_path = os.path.join(base_path, folder_id, f"pred_boundary_points_{folder_id}.npy")
    boundary_points_2_path = os.path.join(base_path, folder_id, f"GT_boundary_points_{folder_id}.npy")

    # 读取点云数据
    boundary_points_1 = load_points(boundary_points_1_path)
    boundary_points_2 = load_points(boundary_points_2_path)

    # 确保数据是三维坐标
    assert boundary_points_1.shape[1] == 3, "Boundary points must be 3D coordinates."
    assert boundary_points_2.shape[1] == 3, "Boundary points must be 3D coordinates."

    # 提取文件名，用于图例
    file_name_1 = os.path.basename(boundary_points_1_path)
    file_name_2 = os.path.basename(boundary_points_2_path)

    # 创建2D图形
    plt.figure(figsize=(10, 10))

    # 可视化两个数据集，使用不同的颜色
    plt.scatter(boundary_points_1[:, 0], boundary_points_1[:, 1], color='b', label=file_name_1, alpha=0.6, s=10)
    plt.scatter(boundary_points_2[:, 0], boundary_points_2[:, 1], color='g', label=file_name_2, alpha=0.6, s=10)

    # 设置一致的坐标轴范围
    min_x = min(np.min(boundary_points_1[:, 0]), np.min(boundary_points_2[:, 0]))
    max_x = max(np.max(boundary_points_1[:, 0]), np.max(boundary_points_2[:, 0]))
    min_y = min(np.min(boundary_points_1[:, 1]), np.min(boundary_points_2[:, 1]))
    max_y = max(np.max(boundary_points_1[:, 1]), np.max(boundary_points_2[:, 1]))

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    # 设置相同的坐标轴比例
    plt.gca().set_aspect('equal', adjustable='box')

    # 添加图例、标题和坐标轴标签
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"2D Visualization of Boundary Points -{folder_id}")

    # 保存图像到对应目录
    output_path = os.path.join(base_path, folder_id, "compare_"+f"{folder_id}.png")
    plt.savefig(output_path)

    # 显示图形
    plt.show()

def main():
    # 遍历文件夹编号并处理
    for folder_id in folder_ids:
        process_folder(folder_id)

if __name__ == "__main__":
    main()
