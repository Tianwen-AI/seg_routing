# 直接选择文件夹中多个文件作为输入
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import time
import random
from connectivity_loss import connect_score
from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision
from thop import profile

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# 将预测结果转化为实际可用的路径
def matrix_to_path(matrix):
    # 存储路径线段
    path_segments = []

    # 按行遍历以找到水平路径
    for y in range(len(matrix)):
        x_start = None
        for x in range(len(matrix[y])):
            if matrix[y][x] == 1:
                if x_start is None:
                    x_start = x  # 记录起点
                if x == len(matrix[y]) - 1 or matrix[y][x + 1] == 0:  # 到达终点
                    if x_start != x:  # 如果起点和终点不同，说明有连续的水平路径
                        # 添加水平线段
                        path_segments.append((x_start, y))
                        path_segments.append((x, y))
                    x_start = None  # 重置起点
            else:
                x_start = None  # 遇到0，重置起点
    # 按列遍历以找到垂直路径
    for x in range(len(matrix[0])):
        y_start = None
        for y in range(len(matrix)):
            if matrix[y][x] == 1:
                if y_start is None:
                    y_start = y  # 记录起点
                if y == len(matrix) - 1 or matrix[y + 1][x] == 0:  # 到达终点
                    if y_start != y:  # 如果起点和终点不同，说明有连续垂直路径
                        # 添加垂直线段
                        path_segments.append((x, y_start))
                        path_segments.append((x, y))

                    y_start = None  # 重置起点
            else:
                y_start = None  # 遇到0，重置起点

    return path_segments

'main'
# 加载 .npy 文件
source_path = './test_input/Source'  # 你要加载的 .npy 文件路径
node_path = './test_input/Nodes'  # 你要加载的 .npy 文件路径

start_time = time.time()
# 加载模型、预测输出
model = torch.load('./model/cg_ukan_new/best_model.pth')
total_length = 0.0
# 遍历第一个文件夹
for file_name in os.listdir(source_path):
    # 检查是否为.npy文件
    if file_name.endswith(".npy"):
        node_file_path = os.path.join(node_path, file_name)
        source_file_path = os.path.join(source_path, file_name)

        souce_0 = np.load(source_file_path)
        nodes_0 = np.load(node_file_path)

        # # 打印文件内容
        # print(data)
        # print("数据形状：", data.shape)
        # print(nodes)

        # start_time = time.time()

        modify_hanan_grid = souce_0[0]
        # print(modify_hanan_grid)
        # # 使用imshow展示
        # plt.imshow(modify_hanan_grid, cmap='viridis', interpolation='none')
        # plt.show()

        grid_size = souce_0[0].shape
        # print('hanan网格尺寸', grid_size)

        # 对模型输入进行resize和归一化
        transform = transforms.Compose([transforms.Resize(size=(256, 256), interpolation=transforms.InterpolationMode.NEAREST)])
        max_pixel_value = 5 # 使用3-5的值对hanan网格进行表示
        source_tensor = transform(torch.from_numpy(np.expand_dims(modify_hanan_grid, axis=0))) / max_pixel_value
        souce_0 = source_tensor.unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(souce_0)

        output = torch.sigmoid(output)
        output = (output > 0.5).float() # 设置阈值进行输出

        # 对输出进行resize
        TRANSFORM_RESIZE = transforms.Compose([transforms.Resize(size=grid_size, interpolation=transforms.InterpolationMode.BILINEAR)])
        # output_np = (TRANSFORM_RESIZE(output).cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).astype(int) # 输出预测相关衡量系数时使用
        output_np = (TRANSFORM_RESIZE(output).cpu().squeeze(0).permute(1, 2, 0).numpy()).astype(int).flatten() # 新 (输出打平层)
        output_np = output_np.reshape(grid_size[0], grid_size[1])

        # print(output_np)

        # # 连通性得分检测
        # # print(target_0)
        # coords = [tuple(row) for row in nodes_0 if any(row)]
        # # print(coords)
        # connectivity_score = connect_score(output_np, np.array(coords))
        # print('连通性得分：', connectivity_score)

        # 路径转化和线长计算
        matrix = TRANSFORM_RESIZE(output).tolist()[0][0]
        path = matrix_to_path(matrix)
        wire_length = 0.0
        for k in range(0, len(path), 2):
            wire_length += manhattan_distance(path[k], path[k + 1])

        total_length += wire_length
        print('路径：', path)
        print('布线线长：', wire_length)

        # 布线预测时间
        print('预测时间：', time.time() - start_time)

        # 结果可视化
        plt.imshow(output_np, cmap='viridis', interpolation='none')
        plt.axis('off')
        # plt.show()
        file_path = os.path.join('./results', file_name)
        plt.savefig(file_path + '_' + str(wire_length) + '.png', dpi=200)
        plt.close()
# # 布线预测时间
# print('预测时间：', time.time() - start_time)
# print('布线总线长：', total_length)