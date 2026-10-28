"""RMSNorm 与 LayerNorm 的对比实验（教材 4.2.2 节，图 4.2「RMS归一化缩放效应图」）。"""

import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

torch.manual_seed(42)


class LayerNorm(nn.Module):
    def __init__(self, dim, epsilon=0.0001):
        super(LayerNorm, self).__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.gamma = torch.nn.Parameter(torch.ones(self.dim))
        self.beta = torch.nn.Parameter(torch.zeros(self.dim))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.epsilon)
        x_out = x_hat * self.gamma + self.beta
        return x_out


class RMSNorm(nn.Module):
    def __init__(self, dim, epsilon=0.0001):
        super(RMSNorm, self).__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.gamma = torch.nn.Parameter(torch.ones(self.dim))

    def forward(self, x):
        RMS = torch.mean(x ** 2.0, dim=-1, keepdim=True)
        x_hat = x / torch.sqrt(RMS + self.epsilon)
        x_out = x_hat * self.gamma
        return x_out


class Arrow3D(FancyArrowPatch):
    """三维坐标轴箭头。"""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)


def draw_axes_arrows(ax, arrow_length=2.0):
    """给三维子图补上带标签的坐标轴箭头与原点标记。"""
    arrow_props = dict(mutation_scale=10, arrowstyle='-|>',
                       color='darkblue', shrinkA=0, shrinkB=0, linewidth=1)

    ax.add_artist(Arrow3D([-arrow_length, arrow_length], [0, 0], [0, 0], **arrow_props))
    ax.text(arrow_length + 0.2, 0, 0, "X", fontsize=14, color='darkblue')

    ax.add_artist(Arrow3D([0, 0], [-arrow_length, arrow_length], [0, 0], **arrow_props))
    ax.text(0, arrow_length + 0.2, 0, "Y", fontsize=14, color='darkblue')

    ax.add_artist(Arrow3D([0, 0], [0, 0], [-arrow_length, arrow_length], **arrow_props))
    ax.text(0, 0, arrow_length + 0.2, "Z", fontsize=14, color='darkblue')

    ax.scatter(0, 0, 0, c='black', s=80, marker='o')
    ax.text(0, 0, -0.5, "O(0,0,0)", fontsize=10, color='black')
    ax.set_proj_type('ortho')
    ax.set_xlabel('X Axis', fontsize=12)
    ax.set_ylabel('Y Axis', fontsize=12)
    ax.set_zlabel('Z Axis', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)


def scatter_norm(ax, x_proj, x_ln, x_rms, tag=""):
    """把原始数据、LayerNorm 输出、RMSNorm 输出画在同一组三维坐标里。"""
    ax.scatter(x_proj[:, 0].tolist(), x_proj[:, 1].tolist(), x_proj[:, 2].tolist(),
               c='green', s=15, alpha=0.8, marker='o', label=f'x{tag}')
    ax.scatter(x_ln[:, 0].tolist(), x_ln[:, 1].tolist(), x_ln[:, 2].tolist(),
               c='blue', s=15, alpha=0.8, marker='^', label=f'LayerNorm(x{tag})')
    ax.scatter(x_rms[:, 0].tolist(), x_rms[:, 1].tolist(), x_rms[:, 2].tolist(),
               c='red', s=15, alpha=0.8, marker='*', label=f'RMSNorm(x{tag})')
    # 图例只收这三条曲线；原点标记在下面补，不进图例（与教材图 4.2 一致）
    ax.legend(loc='upper right')
    draw_axes_arrows(ax)


def main():
    bs = 100
    dim = 3
    linear = nn.Linear(3, 3, bias=False)
    ln = LayerNorm(dim)
    rms = RMSNorm(dim)

    x = torch.randn(bs, dim) + 0.003
    x_proj = linear(x)

    x_ln = ln(x_proj)
    x_rms = rms(x_proj)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    scatter_norm(ax, x_proj, x_ln, x_rms)

    # 第 1、2 维取绝对值，得到结构化数据 X'
    x_proj[:, 0] = x_proj[:, 0].abs()
    x_proj[:, 1] = x_proj[:, 1].abs()

    x_abs_ln = ln(x_proj)
    x_abs_rms = rms(x_proj)

    ax = fig.add_subplot(122, projection='3d')
    scatter_norm(ax, x_proj, x_abs_ln, x_abs_rms, tag="'")

    plt.tight_layout()
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "output")
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, os.path.basename(__file__).replace(".py", ".pdf"))
    plt.savefig(out)

    print("输入形状:", tuple(x.shape))
    print("原始数据均值:", x_proj.mean(dim=0))
    print("绝对值 LayerNorm 均值:", x_abs_ln.mean(dim=0))
    print("绝对值 RMSNorm 均值:", x_abs_rms.mean(dim=0))


main()
