"""RoPE 高低频分组的点积衰减（教材 4.3.3 节，图「高低频位置编码的点积衰减特性分析」）。"""

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
    d_model = 4096
    max_len = 16000
    base = 10000

    rope = RotaryPositionalEncoding(d_model, base, max_len)
    linear = nn.Linear(d_model, d_model)

    x_i = torch.randn(1, d_model) / d_model
    x_i = linear(x_i)
    x_j = torch.randn(1, d_model) / d_model
    x_j = linear(x_j)

    x_i_rot = rope.rotate_vector(x_i, 0)

    rot_products = []
    rot_low_products = []
    rot_high_products = []
    for j in range(1, max_len):
        x_j_rot = rope.rotate_vector(x_j, j)

        rot_products.append(torch.dot(x_i_rot, x_j_rot).item())
        # 低维高频
        rot_low_products.append(torch.dot(x_i_rot[:d_model // 2], x_j_rot[:d_model // 2]).item())
        # 高维低频
        rot_high_products.append(torch.dot(x_i_rot[d_model // 2:], x_j_rot[d_model // 2:]).item())

    rot_products = torch.tensor(rot_products)
    rot_low_products = torch.tensor(rot_low_products)
    rot_high_products = torch.tensor(rot_high_products)

    plt.figure(figsize=(10, 3.5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, max_len), rot_products)
    plt.title('Full-Rank Position Encoding Decay')
    plt.xlabel('Relative Distance')
    plt.ylabel('Inner Product')
    plt.grid(True)
    plt.xlim(-500, 16000)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, max_len), rot_low_products, label='Low-dim (High Freq)')
    plt.plot(range(1, max_len), rot_high_products, label='High-dim (Low Freq)')
    plt.title('Decay of High/Low Frequency Groups')
    plt.xlabel('Relative Distance')
    plt.ylabel('Inner Product')
    plt.legend()
    plt.grid(True)
    plt.xlim(-500, 16000)

    plt.tight_layout()
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "output")
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, os.path.basename(__file__).replace(".py", ".pdf"))
    plt.savefig(out)

    return rot_products, rot_low_products, rot_high_products


rot_products, low, high = plot_position_encoding_similarity()

print("样本数:", rot_products.shape)
print("全维度末段平均绝对值:", rot_products[-1000:].abs().mean())
print("低维组末段平均绝对值:", low[-1000:].abs().mean())
print("高维组末段平均绝对值:", high[-1000:].abs().mean())
