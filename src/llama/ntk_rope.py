"""NTK-RoPE（教材 4.3.3 节，图「放大基数对位置编码向量影响」）。"""

import os

import torch
import matplotlib.pyplot as plt

# 创建 x 值范围
x = torch.linspace(0, 256, 1000)
pos_1 = 4
pos_2 = 8
b = 10000
b_10 = 100000
d = 512

# 计算不同位置与不同底数下的正弦函数
y1 = torch.sin(pos_1 * b ** (-2 * (x - 2) / d))
y2 = torch.sin(pos_2 * b ** (-2 * (x - 2) / d))
y4 = torch.sin(pos_2 * b_10 ** (-2 * (x - 2) / d))

plt.figure(figsize=(10, 3.5))
plt.plot(x, y1, label='pos=4, b=10,000', linewidth=2)
plt.plot(x, y2, label='pos=8, b=10,000', linewidth=2)
plt.plot(x, y4, label='pos=8, b=100,000', linewidth=2)

plt.grid(True)
plt.title('NTK-RoPE')
plt.xlabel('Dimension')
plt.ylabel('Value')
plt.xlim(0, 256)
plt.legend()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "output")
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, os.path.basename(__file__).replace(".py", ".pdf"))
plt.savefig(out)

print("x 形状:", tuple(x.shape))
print("pos=4 与 pos=8,b=100k 在全维度的最大偏差:", (y1 - y4).abs().max())
print("低维区间 (0~64) 的最大偏差:", (y1[:250] - y4[:250]).abs().max())
print("高维区间 (192~256) 的最大偏差:", (y1[750:] - y4[750:]).abs().max())
