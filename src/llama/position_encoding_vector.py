"""位置编码向量随维度的变化（教材 4.3.3 节，图「位置编码向量随维度的变化图」）。"""

import os

import torch
import matplotlib.pyplot as plt

# 创建 x 值范围
x = torch.linspace(0, 256, 1000)
pos_1 = 1
pos_2 = 4
pos_3 = 16
b = 10000
d = 512

# 计算三个不同位置的正弦函数
y1 = torch.sin(pos_1 * b ** (-2 * (x - 2) / d))
y2 = torch.sin(pos_2 * b ** (-2 * (x - 2) / d))
y3 = torch.sin(pos_3 * b ** (-2 * (x - 2) / d))

# 创建图形
plt.figure(figsize=(10, 3.5))
plt.plot(x, y1, label='pos=1, d=512', linewidth=2)
plt.plot(x, y2, label='pos=4, d=512', linewidth=2)
plt.plot(x, y3, label='pos=16, d=512', linewidth=2)

plt.grid(True)
plt.title('Position Encoding Vector vs Dimension')
plt.xlabel('Dimension')
plt.ylabel('Value')
plt.xlim((0, 256))
plt.ylim((-1.1, 1.1))
plt.legend()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "output")
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, os.path.basename(__file__).replace(".py", ".pdf"))
plt.savefig(out)

print("x 形状:", tuple(x.shape))
print("位置 1 在维度 0/128/255 处的值:", y1[0], y1[500], y1[999])
print("位置 16 在维度 0/128/255 处的值:", y3[0], y3[500], y3[999])
