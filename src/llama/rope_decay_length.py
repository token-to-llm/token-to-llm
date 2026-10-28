"""RoPE 在 1k 与 64k 两种上下文长度下的点积衰减（教材 4.3.3 节长文本小节）。"""

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
        x_shift[::2] = -x[1::2]
        x_shift[1::2] = x[::2]

        rope_x = x * cos + x_shift * sin
        return rope_x


def plot_position_encoding_similarity():
    # 参数设置
    d_model = 4096
    max_len = 64000
    base = 10000

    # 左图用短上下文，右图用长上下文
    short_rope = RotaryPositionalEncoding(d_model, base, 1024)
    rope = RotaryPositionalEncoding(d_model, base, max_len)

    linear = nn.Linear(d_model, d_model)

    x_i = torch.randn(1, d_model) / d_model
    x_i = linear(x_i)
    x_j = torch.randn(1, d_model) / d_model
    x_j = linear(x_j)

    # 左：1024 上下文
    x_i_rot = short_rope.rotate_vector(x_i, 0)
    abs_products = []
    for j in range(1, 1024):
        x_j_rot = short_rope.rotate_vector(x_j, j)
        abs_products.append(torch.dot(x_i_rot, x_j_rot).item())

    # 右：64k 上下文
    x_i_rot = rope.rotate_vector(x_i, 0)
    rot_products = []
    for j in range(1, max_len):
        x_j_rot = rope.rotate_vector(x_j, j)
        rot_products.append(torch.dot(x_i_rot, x_j_rot).item())

    abs_products = torch.tensor(abs_products)
    rot_products = torch.tensor(rot_products)

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    plt.grid(True)
    plt.plot(range(1, 1024), abs_products)
    plt.title('RoPE, 1024')
    plt.xlabel('Relative Distance')
    plt.ylabel('Inner Product')
    plt.xlim(-20, 1024)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, max_len), rot_products)
    plt.title('RoPE, 64k')
    plt.xlabel('Relative Distance')
    plt.ylabel('Inner Product')
    plt.grid(True)
    plt.xlim(-1000, 64000)

    plt.tight_layout()
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "output")
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, os.path.basename(__file__).replace(".py", ".pdf"))
    plt.savefig(out)

    return abs_products, rot_products


abs_products, rot_products = plot_position_encoding_similarity()

print("1024 上下文的样本数:", abs_products.shape)
print("64k 上下文的样本数:", rot_products.shape)
print("1024 内积标准差:", abs_products.std())
print("64k 内积标准差:", rot_products.std())
print("64k 末段平均绝对值:", rot_products[-1000:].abs().mean())
