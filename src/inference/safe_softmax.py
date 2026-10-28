"""Safe-Softmax（教材 10.2 节 · 代码清单「Safe-Softmax代码」）。"""

import torch

X = torch.tensor([-0.3, 0.2, 0.5, 0.7, 0.1, 0.8])
X_max = X.max()
X_exp_sum_sub_max = torch.exp(X - X_max).sum()
X_safe_softmax = torch.exp(X - X_max) / X_exp_sum_sub_max

print("输入:", X)
print("最大值:", X_max)
print("Safe-Softmax 结果:", X_safe_softmax)
