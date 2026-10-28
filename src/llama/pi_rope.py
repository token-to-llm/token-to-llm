"""位置内插 RoPE 与原始 RoPE 的衰减对比（教材 4.3.3 节，图「位置编码内插」）。"""

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
    """原始旋转位置编码。"""

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
    max_len = 64000
    base = 10000

    pi_rope = PI_RotaryPositionalEncoding(d_model, base, max_len, 2)
    rope = RotaryPositionalEncoding(d_model, base, max_len)
    linear = nn.Linear(d_model, d_model)

    x_i = torch.randn(1, d_model) / d_model
    x_i = linear(x_i)
    x_j = torch.randn(1, d_model) / d_model
    x_j = linear(x_j)

    pi_products = []
    rot_products = []
    for j in range(1, max_len):
        pi_products.append(torch.dot(pi_rope.rotate_vector(x_i, 0),
                                     pi_rope.rotate_vector(x_j, j)).item())
        rot_products.append(torch.dot(rope.rotate_vector(x_i, 0),
                                      rope.rotate_vector(x_j, j)).item())

    pi_products = torch.tensor(pi_products)
    rot_products = torch.tensor(rot_products)

    plt.figure(figsize=(8, 3.5))
    plt.subplot(1, 1, 1)
    plt.plot(range(1, max_len), pi_products, label='PI RoPE, s=2')
    plt.plot(range(1, max_len), rot_products, label='RoPE, s=1')
    plt.title('Position Interpolation on RoPE')
    plt.xlabel('Relative Distance')
    plt.ylabel('Inner Product')
    plt.legend()
    plt.xlim(-1000, 64000)
    plt.grid(True)

    plt.tight_layout()
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "output")
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, os.path.basename(__file__).replace(".py", ".pdf"))
    plt.savefig(out)

    return pi_products, rot_products


pi_products, rot_products = plot_position_encoding_similarity()

print("样本数:", pi_products.shape)
print("PI 末段平均绝对值:", pi_products[-1000:].abs().mean())
print("RoPE 末段平均绝对值:", rot_products[-1000:].abs().mean())
