"""块状 Online-Softmax（教材 10.2 节 · 代码清单「块状Onlie-Softmax代码」）。"""

import torch

X = torch.tensor([-0.3, 0.2, 0.5, 0.7, 0.1, 0.8])
X_block = torch.split(X, split_size_or_sections=3, dim=0)
m_0 = X_block[0].max()
l_0 = torch.exp(X_block[0] - m_0).sum()
m_1 = X_block[1].max()
l_1 = torch.exp(X_block[1] - m_1).sum()

m_new = torch.max(m_0, m_1)
l_new = l_0 * torch.exp(m_0 - m_new) + l_1 * torch.exp(m_1 - m_new)
X_result = torch.exp(X - m_new) / l_new

print("分块数量:", len(X_block))
print("各块最大值:", m_0, m_1)
print("合并后统计量:", m_new, l_new)
print("块状 Online-Softmax 结果:", X_result)
