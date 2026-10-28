"""损失函数 $-\log \sigma(x)$ 的图像（教材 8.3.2 节 BT 模型，图「损失函数图像」）。"""

import os

import torch
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def neg_log_sigmoid(x):
    return -torch.log(sigmoid(x))

# 准备数据
x = torch.linspace(-10, 10, 500)

plt.figure(figsize=(6, 3))
plt.plot(x, sigmoid(x), label='Sigmoid: σ(x)')
plt.plot(x, neg_log_sigmoid(x), label='-log σ(x)')

plt.title('Loss Function -log(sigmoid(x))')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.xlim(-4, 4)
plt.ylim(0, 4)
plt.tight_layout()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "output")
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, os.path.basename(__file__).replace(".py", ".pdf"))
plt.savefig(out)

print("x 形状:", tuple(x.shape))
print("σ(0) 与 -logσ(0):", sigmoid(torch.tensor(0.0)), neg_log_sigmoid(torch.tensor(0.0)))
print("x=4 处的损失:", neg_log_sigmoid(torch.tensor(4.0)))
print("x=-4 处的损失:", neg_log_sigmoid(torch.tensor(-4.0)))
