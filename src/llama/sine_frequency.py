"""正弦函数的频率与周期（教材 4.3.3 节 RoPE 小节，图「正弦函数周期与频率的关系」）。"""

import os

import torch
import matplotlib.pyplot as plt

# 创建 x 值范围
x = torch.linspace(0, 10, 1000)

# 计算三个不同频率的正弦函数
y1 = torch.sin(0.5 * x)
y2 = torch.sin(x)
y3 = torch.sin(2 * x)

# 创建图形
plt.figure(figsize=(10, 3.5))
plt.plot(x, y1, label='sin(0.5x)', linewidth=2)
plt.plot(x, y2, label='sin(x)', linewidth=2)
plt.plot(x, y3, label='sin(2x)', linewidth=2)

plt.grid(True)
plt.title('Sine Functions with Different Frequencies')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim((0, 10))
plt.ylim((-1.1, 1.1))
plt.legend()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "output")
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, os.path.basename(__file__).replace(".py", ".pdf"))
plt.savefig(out)

print("x 形状:", tuple(x.shape))
print("周期 4pi 的正弦在 x=10 处的值:", y1[-1])
print("周期 2pi 的正弦在 x=10 处的值:", y2[-1])
print("周期 pi 的正弦在 x=10 处的值:", y3[-1])
