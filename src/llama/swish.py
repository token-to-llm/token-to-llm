"""Swish 与主流激活函数对比（教材 4.2.1 节 SwiGLU / Swish 小节）。"""

import math
import os

import torch
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def relu(x):
    return torch.maximum(torch.zeros_like(x), x)


def gelu(x):
    # GELU 近似实现
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

# 创建数据
x = torch.linspace(-6, 6, 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

# 左侧子图：Swish 的构造
ax1.plot(x, x, label='y=x', linestyle='--', color='blue', linewidth=2)
ax1.plot(x, sigmoid(x), label='sigmoid(x)', color='red', linewidth=2)
ax1.plot(x, x * sigmoid(x), label='Swish(x)', color='green', linewidth=2)
ax1.grid(True)
ax1.set_title('Swish Activation')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_xlim((-6, 6))
ax1.legend()
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# 右侧子图：主流激活函数
ax2.plot(x, relu(x), label='ReLU', color='purple', linewidth=2)
ax2.plot(x, gelu(x), label='GELU', color='orange', linewidth=2)
ax2.plot(x, x * sigmoid(x), label='Swish', color='green', linewidth=2)
ax2.grid(True)
ax2.set_title('Mainstream Activation Functions')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_xlim((-6, 6))
ax2.legend()
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "output")
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, os.path.basename(__file__).replace(".py", ".pdf"))
plt.savefig(out)

print("x 形状:", tuple(x.shape))
print("Swish 在 0 点的值:", (x * sigmoid(x))[500])
print("Swish 最小值:", (x * sigmoid(x)).min())
print("ReLU 与 GELU 在 x=-3 处的值:", relu(torch.tensor(-3.0)), gelu(torch.tensor(-3.0)))
