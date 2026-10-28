"""多块状 Online-Softmax（教材 10.2 节 · 代码清单「多块状Onlie-Softmax代码」）。"""

import torch

X = torch.tensor([-0.3, 0.2, 0.5, 0.7, 0.1, 0.8])
X_blocks = torch.split(X, split_size_or_sections=2, dim=0)
M_new = torch.tensor([-float('inf')])
L_new = torch.tensor([0.0])

M_list = []
L_list = []
for i in range(len(X_blocks)):
    m = torch.max(X_blocks[i])
    l = torch.exp(X_blocks[i] - m).sum()
    M_list.append(m)
    L_list.append(l)

for i in range(len(X_blocks)):
    M_old = M_new
    L_old = L_new
    # update M, L
    M_new = torch.max(M_new, M_list[i])
    L_new = L_old * torch.exp(M_old - M_new) + L_list[i] * torch.exp(M_list[i] - M_new)

X_result = torch.exp(X - M_new) / L_new

print("分块数量:", len(X_blocks))
print("各块最大值:", torch.stack(M_list))
print("合并后统计量:", M_new, L_new)
print("多块状 Online-Softmax 结果:", X_result)
