"""Online-Softmax（教材 10.2 节 · 代码清单「Online-Softmax代码」）。"""

import torch

X = torch.tensor([-0.3, 0.2, 0.5, 0.7, 0.1, 0.8])
X_pre = X[:-1]  # tensor([-0.3000,  0.2000,  0.5000,  0.7000,  0.1000])
X_max_pre = X_pre.max()
X_sum_pre = torch.exp(X_pre - X_max_pre).sum()
X_max_cur = torch.max(X_max_pre, X[-1])  # X[-1] = 0.8
X_sum_cur = X_sum_pre * torch.exp(X_max_pre - X_max_cur) + torch.exp(X[-1] - X_max_cur)
X_online_softmax = torch.exp(X - X_max_cur) / X_sum_cur

print("前 5 个元素的最大值:", X_max_pre)
print("前 5 个元素的指数和:", X_sum_pre)
print("Online-Softmax 结果:", X_online_softmax)
