"""Softmax（教材 10.2 节 · 代码清单「Softmax代码」）。"""

import torch

X = torch.tensor([-0.3, 0.2, 0.5, 0.7, 0.1, 0.8])
X_exp_sum = X.exp().sum()
X_softmax = torch.exp(X) / X_exp_sum

print("输入:", X)
print("指数和:", X_exp_sum)
print("Softmax 结果:", X_softmax)
