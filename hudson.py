import pyrocko.plot.hudson as hudson
import matplotlib.pyplot as plt
from pyrocko.plot.beachball import plot_beachball_mpl
import numpy as np
def ae_uvt(TT, KK):
    TAU = TT * (1 - np.abs(KK))
    UU = np.full_like(TT, np.nan)
    VV = np.full_like(TT, np.nan)

    # 2nd and 4th quadrants
    II = (TAU > 0) & (KK < 0) | (TAU < 0) & (KK > 0)
    UU[II] = TAU[II]
    VV[II] = KK[II]

    # First quadrant, Region A
    II = (TAU < 4 * KK) & (TAU >= 0) & (KK >= 0)
    UU[II] = TAU[II] / (1 - TAU[II] / 2)
    VV[II] = KK[II] / (1 - TAU[II] / 2)

    # First quadrant, Region B
    II = (TAU >= 4 * KK) & (TAU >= 0) & (KK >= 0)
    UU[II] = TAU[II] / (1 - 2 * KK[II])
    VV[II] = KK[II] / (1 - 2 * KK[II])

    # Third quadrant
    II = (TAU >= 4 * KK) & (TAU <= 0) & (KK <= 0)
    UU[II] = TAU[II] / (1 + TAU[II] / 2)
    VV[II] = KK[II] / (1 + TAU[II] / 2)

    II = (TAU < 4 * KK) & (TAU <= 0) & (KK <= 0)
    UU[II] = TAU[II] / (1 + 2 * KK[II])
    VV[II] = KK[II] / (1 + 2 * KK[II])

    if np.any(np.isnan(VV)):
        raise ValueError("Error plotting point in [uu, vv] space.")

    return UU, VV

def plot_beachball_from_matrices(matrices):
    """
    绘制海滩球图，并将每个矩张量投影到二维坐标系中
    
    参数:
    matrices (list of lists): 包含多个矩张量的二维列表，每个矩张量为6个元素的元组
    
    返回:
    None
    """
    # 创建图形和坐标轴
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    
    # 绘制坐标轴
    hudson.draw_axes(ax)
    
    # 创建 VV 数组
    VV = np.arange(-1, 1.1, 0.1)

    for i in range(10):
        x, y = ae_uvt(VV, VV[i] * np.ones_like(VV))
        plt.plot(x, y, 'k--',linewidth=0.5)

        x, y = ae_uvt(VV, -VV[i] * np.ones_like(VV))
        plt.plot(x, y, 'k--',linewidth=0.5)

        x, y = ae_uvt(VV[i] * np.ones_like(VV), VV)
        plt.plot(x, y, 'k--',linewidth=0.5)

        x, y = ae_uvt(-VV[i] * np.ones_like(VV), VV)
        plt.plot(x, y, 'k--',linewidth=0.5)

    # 遍历矩阵列表
    for matrix in matrices:
        # 获取project函数返回的坐标
        coordinates = hudson.project(tuple(matrix))  # 将矩张量元组传递给project函数
        
        # 假设返回的是一个二维坐标，可以这样绘制
        x, y = coordinates
        
        # 绘制海滩球图
        plot_beachball_mpl(tuple(matrix), ax, position=(x, y), linewidth=0.5)
    
    # 显示图形
    plt.show()

if __name__ == "__main__":
    # 使用示例
    matrices = [
        (1, 2, 3, 4, 5, 6),
        (1, 1, 1, 1, 1, 1),  # 示例中的其他矩张量
        (1, 2, 3, 4, 5, 6)
    ]
    plot_beachball_from_matrices(matrices)
