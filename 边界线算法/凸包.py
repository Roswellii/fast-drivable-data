import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# 假设 p1 和 p2 是地面和人行道的点云
p1 = np.random.rand(100, 2)  # 地面点云
p2 = np.random.rand(100, 2) + np.array([2, 0])  # 人行道点云，平移

# 合并两簇点
points = np.vstack((p1, p2))

# 计算点云之间的最近邻
nbrs1 = NearestNeighbors(n_neighbors=1).fit(p1)
distances1, indices1 = nbrs1.kneighbors(p2)

nbrs2 = NearestNeighbors(n_neighbors=1).fit(p2)
distances2, indices2 = nbrs2.kneighbors(p1)

# 找到最小距离的点对
boundary_points_p1 = p1[indices1.flatten()]
boundary_points_p2 = p2[indices2.flatten()]

# 可视化交线
plt.scatter(p1[:, 0], p1[:, 1], label='Ground Points', alpha=0.6)
plt.scatter(p2[:, 0], p2[:, 1], label='Sidewalk Points', alpha=0.6)
plt.scatter(boundary_points_p1[:, 0], boundary_points_p1[:, 1], color='r', label='Boundary Points from Ground')
plt.scatter(boundary_points_p2[:, 0], boundary_points_p2[:, 1], color='g', label='Boundary Points from Sidewalk')

# 连接边界点，显示交线
for i in range(len(boundary_points_p1)):
    plt.plot([boundary_points_p1[i, 0], boundary_points_p2[i, 0]],
             [boundary_points_p1[i, 1], boundary_points_p2[i, 1]], 'k--', lw=0.5)

plt.legend()
plt.show()
