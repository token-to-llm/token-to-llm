"""位置内插 PI（教材 4.3.3 节，图「位置编码内插前后对比」）。"""

import os

import torch
import matplotlib.pyplot as plt

# 创建 x 值范围
x = torch.linspace(0, 256, 1000)
pos_1 = 8
pos_2 = 16
pos_3 = 17
b = 10000
d = 512

# 计算三个不同位置的正弦函数
y1 = torch.sin(pos_1 * b ** (-2 * (x - 2) / d))
y2 = torch.sin(pos_2 * b ** (-2 * (x - 2) / d))
y3 = torch.sin((pos_3 / 2.0) * b ** (-2 * (x - 2) / d))

plt.figure(figsize=(10, 3.5))
plt.plot(x, y1, label='pos=8, s=1', linewidth=2)
plt.plot(x, y2, label='pos=16, s=1', linewidth=2)
plt.plot(x, y3, label='pos=17, s=2', linewidth=2)

plt.grid(True)
plt.title('Position Interpolation')
plt.xlabel('Dimension')
plt.ylabel('Value')
plt.xlim(0, 256)
plt.ylim(-1.1, 1.1)
plt.legend()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "output")
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, os.path.basename(__file__).replace(".py", ".pdf"))
plt.savefig(out)

print("x 形状:", tuple(x.shape))
print("pos=8 与 pos=17,s=2 的最大偏差:", (y1 - y3).abs().max())
print("pos=8 与 pos=16 的最大偏差:", (y1 - y2).abs().max())
