"""缩放点积注意力（教材 2.3.1 节 · 代码清单「缩放点积注意力 Pytorch 代码」）。"""

import math

import torch
import torch.nn as nn

torch.manual_seed(42)


def ScaledDotProductAttention(Q, K, V, mask):
    B, H, L, D = K.size()
    Kt = K.transpose(2, 3)
    S = Q @ Kt / math.sqrt(D)
    if mask is not None:
        S = S.masked_fill(mask, -torch.tensor(float('inf')))
    P = nn.functional.softmax(S, dim=-1)
    Z = P @ V
    return Z

# 模拟数据
bs = 4  # 批大小
heads = 8  # 注意力头数
seq_len = 16  # 序列长度
dim = 64  # 词嵌入维度
Q = torch.randn(bs, heads, seq_len, dim)
K = torch.randn(bs, heads, seq_len, dim)
V = torch.randn(bs, heads, seq_len, dim)
M = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bool)
out = ScaledDotProductAttention(Q, K, V, M)

print("Q 形状:", tuple(Q.shape))
print("输出形状:", tuple(out.shape))
