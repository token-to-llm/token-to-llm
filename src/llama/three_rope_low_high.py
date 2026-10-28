"""三种位置编码在高频与低频上的衰减（教材 4.3.3 节，图「高低频位置编码的衰减性分析」）。"""

import math
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)


class PI_RotaryPositionalEncoding:
    """带内插系数 s 的旋转位置编码。"""

    def __init__(self, d_model, base=10000, max_len=1024, s=2):
        self.pi_s = s
        self.d_model = d_model
        self.base = base
        self.max_len = max_len

        position = torch.arange(0, max_len * s, dtype=torch.float)
        div_term = (- torch.arange(0, d_model, 2).float() / self.d_model) * math.log(self.base)
        div_term = div_term.exp()

        self.theta = torch.outer(position / self.pi_s, div_term)
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


class RotaryPositionalEncoding:
    """原始旋转位置编码，底数可选。"""

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


def split_products(enc, x_i, x_j, d_model, max_len):
    """返回 (高频组内积, 低频组内积) 两条曲线。"""
    low_products = []
    high_products = []
    for j in range(1, max_len):
        x_i_rot = enc.rotate_vector(x_i, 0)
        x_j_rot = enc.rotate_vector(x_j, j)

        low_products.append(torch.dot(x_i_rot[:d_model // 2], x_j_rot[:d_model // 2]).item())
        high_products.append(torch.dot(x_i_rot[d_model // 2:], x_j_rot[d_model // 2:]).item())
    return torch.tensor(low_products), torch.tensor(high_products)


def plot_position_encoding_similarity():
    d_model = 4096
    max_len = 1024
    base = 10000
    base_10 = 100000

    pi_rope = PI_RotaryPositionalEncoding(d_model, base, max_len, 2)
    rope = RotaryPositionalEncoding(d_model, base, max_len)
    ntk_rope = RotaryPositionalEncoding(d_model, base_10, max_len)
    linear = nn.Linear(d_model, d_model)

    x_i = torch.randn(1, d_model) / d_model
    x_i = linear(x_i)
    x_j = torch.randn(1, d_model) / d_model
    x_j = linear(x_j)

    pi_low, pi_high = split_products(pi_rope, x_i, x_j, d_model, max_len)
    rot_low, rot_high = split_products(rope, x_i, x_j, d_model, max_len)
    ntk_low, ntk_high = split_products(ntk_rope, x_i, x_j, d_model, max_len)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 1, 1)

    plt.plot(range(1, max_len), pi_low, label='Low-freq PI RoPE')
    plt.plot(range(1, max_len), rot_low, label='Low-freq RoPE')
    plt.plot(range(1, max_len), ntk_low, label='Low-freq NTK-RoPE')

    plt.plot(range(1, max_len), pi_high, label='High-freq PI RoPE')
    plt.plot(range(1, max_len), rot_high, label='High-freq RoPE')
    plt.plot(range(1, max_len), ntk_high, label='High-freq NTK-RoPE')

    plt.title('Position Encoding Decay by Frequency Band')
    plt.xlabel('Relative Distance')
    plt.ylabel('Inner Product')
    plt.legend(loc='upper right')
    plt.xlim(-10, 1024)
    plt.grid(True)

    plt.tight_layout()
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "output")
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, os.path.basename(__file__).replace(".py", ".pdf"))
    plt.savefig(out)

    return rot_low, rot_high, ntk_low, ntk_high

rot_low, rot_high, ntk_low, ntk_high = plot_position_encoding_similarity()

print("样本数:", rot_low.shape)
print("RoPE 高频组标准差:", rot_high.std())
print("NTK-RoPE 高频组标准差:", ntk_high.std())
print("RoPE 低频组末段平均绝对值:", rot_low[-100:].abs().mean())
print("NTK-RoPE 低频组末段平均绝对值:", ntk_low[-100:].abs().mean())
