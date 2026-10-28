"""多块状 Online-Softmax（简化版，教材 10.2 节 · 代码清单「多块状Onlie-Softmax简化代码」）。"""

import torch

X = torch.tensor([-0.3, 0.2, 0.5, 0.7, 0.1, 0.8])
X_blocks = torch.split(X, split_size_or_sections=2, dim=0)
M_new = torch.tensor([-float('inf')])
L_new = torch.tensor([0.0])

for i in range(len(X_blocks)):
    M_old = M_new
    L_old = L_new
    # 更新 M, L
    M = torch.max(X_blocks[i])
    M_new = torch.max(M, M_old)
    L_new = L_old * torch.exp(M_old - M_new) + torch.exp(X_blocks[i] - M_new).sum()

X_result = torch.exp(X - M_new) / L_new

print("分块数量:", len(X_blocks))
print("合并后统计量:", M_new, L_new)
print("简化版 Online-Softmax 结果:", X_result)
