"""ReLU 与 GELU 激活函数对比（教材 3.2.2 节 GELU 小节）。"""

import math
import os

import torch
import matplotlib.pyplot as plt

SQRT_2 = math.sqrt(2.0)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))

# 1. ReLU 与 GELU
x = torch.linspace(-3, 3, 1000)
relu = torch.maximum(torch.zeros_like(x), x)
gelu = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

ax1.plot(x, relu, label='ReLU', color='blue')
ax1.plot(x, gelu, label='GELU', color='red', linestyle='--')
ax1.set_title('ReLU vs GELU')
ax1.legend()
ax1.grid(True)
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# 2. 标准正态分布的概率密度函数
x = torch.linspace(-4, 4, 1000)
pdf = torch.exp(-x ** 2 / 2) / math.sqrt(2 * math.pi)

# 找到 CDF = 0.3 对应的 x 值
x_03 = SQRT_2 * torch.erfinv(torch.tensor(2 * 0.3 - 1))

mask = x <= x_03
x_fill = x[mask]
pdf_fill = pdf[mask]

ax2.plot(x, pdf, color='green', label='PDF')
ax2.fill_between(x_fill, pdf_fill, alpha=0.4, color='red', label='Area = 0.3')
ax2.axvline(x=x_03, color='red', linestyle='--', alpha=0.5)
ax2.set_title('Standard Normal PDF')
ax2.grid(True)
ax2.set_xlabel('x')
ax2.set_ylabel('Density')
ax2.legend()

# 3. 标准正态分布的累积分布函数
cdf = 0.5 * (1 + torch.erf(x / SQRT_2))

ax3.plot(x, cdf, color='purple', label='CDF')
ax3.plot([x_03], [0.3], 'ro', label=f'(x={x_03:.3f}, y=0.3)')
ax3.axhline(y=0.3, color='red', linestyle='--', alpha=0.5)
ax3.axvline(x=x_03, color='red', linestyle='--', alpha=0.5)
ax3.set_title('Standard Normal CDF')
ax3.grid(True)
ax3.set_xlabel('x')
ax3.set_ylabel('Cumulative Probability')
ax3.legend()

plt.tight_layout()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "output")
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, os.path.basename(__file__).replace(".py", ".pdf"))
plt.savefig(out)

print("x 形状:", tuple(x.shape))
print("GELU 端点:", gelu[0], gelu[-1])
print("CDF = 0.3 处的分位点:", x_03)
print("该点累积概率:", 0.5 * (1 + torch.erf(x_03 / SQRT_2)))
