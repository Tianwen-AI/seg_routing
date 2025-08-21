from collections import deque
import numpy as np

# BFS 检查连通性 广度优先搜索
def is_connected(matrix, start, targets):
    # 定义方向向量，用于移动 (上、下、左、右)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    rows, cols = len(matrix), len(matrix[0])
    visited = set()
    queue = deque([start])
    visited.add(start)

    connected_count = 0

    while queue:
        current = queue.popleft()
        if current in targets:
            connected_count += 1
            targets.remove(current)  # 已连通点移除

        for dy, dx in directions:
            ny, nx = current[0] + dy, current[1] + dx
            if 0 <= ny < rows and 0 <= nx < cols and (ny, nx) not in visited:
                if matrix[ny][nx] == 1:  # 只沿路径为1的点移动
                    visited.add((ny, nx))
                    queue.append((ny, nx))

    return connected_count

def connect_score(matrix, points):
    connected_points = 0
    coordinates = np.vstack([points[0], points[1:][~np.all(points[1:] == [0, 0, 0], axis=1)]])
    # 转换坐标为二维矩阵中的坐标
    coordinates = [(y, x) for x, y, _ in coordinates]  # 转换成 (y, x) 格式

    for coordinate in coordinates:
        start_point = coordinate
        connected_points_count = is_connected(matrix, start_point, set(coordinates))
        if connected_points_count >= connected_points:
            connected_points = connected_points_count

    return connected_points / len(coordinates)

def connect_loss(matrix, points):
    connected_points = 0
    coordinates = np.vstack([points[0], points[1:][~np.all(points[1:] == [0, 0, 0], axis=1)]])
    # 转换坐标为二维矩阵中的坐标
    coordinates = [(y, x) for x, y, _ in coordinates]  # 转换成 (y, x) 格式

    for coordinate in coordinates:
        start_point = coordinate
        connected_points_count = is_connected(matrix, start_point, set(coordinates))
        if connected_points_count >= connected_points:
            connected_points = connected_points_count

    return -np.log((connected_points / len(coordinates)) + 1e-6) # 使用对数构建损失（突出x越大越差的非线性关系）



