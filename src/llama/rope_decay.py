"""RoPE 位置编码点积衰减曲线（教材 4.3.3 节，图「RoPE 位置编码点积衰减曲线图」）。"""

import math
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)


class RotaryPositionalEncoding:
    def __init__(self, d_model, base=10000, max_len=1024):
        self.d_model = d_model
        self.base = base
        self.max_len = max_len

        # 创建旋转矩阵
        position = torch.arange(0, max_len, dtype=torch.float)
        div_term = (- torch.arange(0, d_model, 2).float() / self.d_model) * math.log(self.base)
        div_term = div_term.exp()

        self.theta = torch.outer(position, div_term)
        self.cos = self.theta.cos()
        self.sin = self.theta.sin()

    def rotate_vector(self, x, pos):
        x = x[0]

        cos = self.cos[pos]
        cos = torch.repeat_interleave(cos, 2)
        sin = self.sin[pos]
        sin = torch.repeat_interleave(sin, 2)

        x_shift = torch.zeros_like(x)
        x_shift[::2] = -x[1::2]  # 偶数位置放入负的奇数位置元素
        x_shift[1::2] = x[::2]   # 奇数位置放入偶数位置元素

        rope_x = x * cos + x_shift * sin
        return rope_x


def plot_position_encoding_similarity():
    # 参数设置
    d_model = 4096
    max_len = 16000
    base = 10000

    rope = RotaryPositionalEncoding(d_model, base, max_len)
    linear = nn.Linear(d_model, d_model)

    # 位置 0 与位置 j 的随机向量
    x_i = torch.randn(1, d_model) / d_model
    x_i = linear(x_i)
    x_j = torch.randn(1, d_model) / d_model
    x_j = linear(x_j)

    x_i_rot = rope.rotate_vector(x_i, 0)

    # 计算不同位置 j 的内积
    rot_products = []
    for j in range(1, max_len):
        x_j_rot = rope.rotate_vector(x_j, j)
        rot_prod = torch.dot(x_i_rot, x_j_rot)
        rot_products.append(rot_prod.item())

    rot_products = torch.tensor(rot_products)

    # 版式与教材图 4.13 一致：8×7 画布，曲线画在下半幅
    plt.figure(figsize=(8, 7))
    plt.subplot(2, 1, 2)
    plt.plot(range(1, max_len), rot_products)
    plt.title('RoPE Inner Product Decay')
    plt.xlabel('Relative Distance')
    plt.ylabel('Inner Product')
    plt.grid(True)
    plt.xlim(-500, 16000)

    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "output")
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, os.path.basename(__file__).replace(".py", ".pdf"))
    plt.savefig(out)

    return rot_products


rot_products = plot_position_encoding_similarity()

print("相对距离样本数:", 16000 - 1)
print("前 10 步内积:", rot_products[:10])
print("内积最大值:", rot_products.max(), "最小值:", rot_products.min())
