import numpy as np
import matplotlib.pyplot as plt
from pyrocko import moment_tensor
from matplotlib.colors import LinearSegmentedColormap

def plot_moment_tensor_diamond(M_matrix):
    """
    绘制矩张量的钻石图，其中输入矩张量的值为二维矩阵，
    每行包含M1-M6的值（每个事件的矩张量值）。
    
    参数:
    - M_matrix: 一个二维数组，其中每行包含6个矩张量元素M1, M2, M3, M4, M5, M6
    """
    N_events = M_matrix.shape[0]  # 获取事件的数量
    
    dc = np.zeros(N_events)
    clvd = np.zeros(N_events)
    iso = np.zeros(N_events)

    # 处理每个事件的矩张量数据
    for i in range(N_events):
        M = np.array([[ M_matrix[i, 0], M_matrix[i, 1], M_matrix[i, 2]],
                      [ M_matrix[i, 1], M_matrix[i, 3], M_matrix[i, 4]],
                      [ M_matrix[i, 2], M_matrix[i, 4], M_matrix[i, 5]]])
        
        # 使用MomentTensor计算DC, CLVD和ISO
        mt = moment_tensor.MomentTensor(M)
        res = mt.standard_decomposition()

        M1, M2, M3 = np.linalg.eigvals(M)  # 获取特征值M1, M2, M3
        sign_clvd = 1 if M1 + M3 - 2 * M2 > 0 else -1
        sign_iso = 1 if np.trace(M) > 0 else -1

        dc[i], clvd[i], iso[i] = res[0][1], res[1][1] * sign_clvd, res[2][1] * sign_iso

    # 创建钻石图（ISO vs CLVD）
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.axis('off')

    # 定义钻石形状
    diamond_x = np.array([-1, 0, 1, 0, -1])
    diamond_y = np.array([0, -1, 0, 1, 0])

    # 定义颜色深度的尺度（这里的深度为DC）
    N_levels = 10  # 颜色层次
    DC_max = 1.0  # 最大DC
    DC_min = 0  # 最小DC

    # 定义自定义颜色映射：中间为(25, 25, 225)，0.1为(229, 229, 255)
    colors = [(0.0, "white"),  
              (0.1, (229/255, 229/255, 255/255)),  
              (0.9, (25/255, 25/255, 225/255)),   
              (1.0, (25/255, 25/255, 225/255))]
    cmap = LinearSegmentedColormap.from_list("custom_blue", colors)

    # 绘制钻石形状（自定义颜色渐变表示双力偶DC）
    for i_level in range(N_levels):
        scale = (N_levels - i_level) / N_levels
        ax.fill(scale * diamond_x, scale * diamond_y, color=cmap(i_level / N_levels))

    # 绘制坐标轴
    ax.plot([-1, 1], [0, 0], 'k', linewidth=0.8)
    ax.plot([0, 0], [-1, 1], 'k', linewidth=0.8)
    ax.plot(diamond_x, diamond_y, 'k', linewidth=1.0)

    # 调整ISO和CLVD的缩放比例
    scale_iso_clvd = 1.0
    scaled_iso = scale_iso_clvd * iso
    scaled_clvd = scale_iso_clvd * clvd

    # 绘制事件点
    ax.plot(scaled_clvd, scaled_iso, 'k.', markersize=5)

    # 添加横向颜色条，表示双力偶（DC）深度
    norm = plt.Normalize(DC_min, DC_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # 必须设置这个，否则会报错

    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal')  # 显式指定图形中的轴
    cbar.set_label('双力偶部分 (DC)')

    # 添加 CLVD 和 ISO 轴标签
    ax.text(1.2, 0, 'CLVD', fontsize=12, ha='center', va='center', rotation=0)
    ax.text(0, 1.1, 'ISO', fontsize=12, ha='center', va='center', rotation=0)

    # 显示图像
    plt.show()

if __name__ == "__main__":
    # 示例调用
    # 假设M_matrix为一个2D矩阵，其中每行包含6个M1-M6的值
    M_matrix = np.random.uniform(-1, 1, (1000, 6))  # 假设模拟1000个事件的M1-M6
    plot_moment_tensor_diamond(M_matrix)
