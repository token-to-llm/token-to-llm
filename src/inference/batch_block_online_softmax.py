"""批量块状 Online-Softmax（教材 10.2 节 · 代码清单「批量块状Onlie-Softmax代码」）。"""

import torch

torch.manual_seed(42)

X_batch = torch.randn(4, 6)
X_blocks = torch.split(X_batch, 2, dim=1)
b, _ = X_batch.shape
M_new = torch.ones(b, 1) * torch.tensor([-float('inf')])
L_new = torch.zeros(b, 1)
for X_block in X_blocks:
    M_old = M_new
    L_old = L_new
    M, _ = torch.max(X_block, dim=1, keepdim=True)
    M_new = torch.maximum(M, M_old)
    L_new = L_old * torch.exp(M_old - M_new) + torch.exp(X_block - M_new).sum(dim=1, keepdim=True)

X_result = torch.exp(X_batch - M_new) / L_new

print("输入形状:", tuple(X_batch.shape))
print("分块数量:", len(X_blocks))
print("合并后统计量形状:", tuple(M_new.shape), tuple(L_new.shape))
print("批量 Online-Softmax 结果:", X_result)
print("每行和:", X_result.sum(dim=1))
