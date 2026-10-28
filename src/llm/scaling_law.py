"""幂律与逆幂律曲线（教材 1.3.2 节「缩放定律」，对应图 1-x）。"""

import os

import torch
import matplotlib.pyplot as plt

x = torch.linspace(0.1, 5, 1000)
p = 0.5

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

# y = x^p
y1 = x ** p
ax1.plot(x, y1)
ax1.set_title('Power Law')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)

# y = (1/x)^p
y2 = (1 / x) ** p
ax2.plot(x, y2)
ax2.set_title('Inverse Power Law')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(True)

# 双对数坐标下的逆幂律
ax3.loglog(x, y2)
ax3.set_title('Inverse Power Law (Log-Log)')
ax3.set_xlabel('log(x)')
ax3.set_ylabel('log(y)')
ax3.grid(True)

plt.tight_layout()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "output")
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, os.path.basename(__file__).replace(".py", ".pdf"))
plt.savefig(out)

print("x 形状:", tuple(x.shape))
print("y = x^0.5 端点:", y1[0], y1[-1])
print("y = x^-0.5 端点:", y2[0], y2[-1])
