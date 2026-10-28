"""多头注意力机制（教材 2.3.2 节 · 代码清单「多头注意力机制 Pytorch 代码」）。"""

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


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.dim = dim
        self.WQ = nn.Linear(dim, dim)
        self.WK = nn.Linear(dim, dim)
        self.WV = nn.Linear(dim, dim)
        self.WO = nn.Linear(dim, dim)

    def forward(self, X, mask):
        B, L, D = X.size()
        head_dim = self.dim // self.heads  # 单头词向量维度
        Q, K, V = self.WQ(X), self.WK(X), self.WV(X)
        Q = Q.reshape(B, L, self.heads, head_dim).transpose(1, 2)
        K = K.reshape(B, L, self.heads, head_dim).transpose(1, 2)
        V = V.reshape(B, L, self.heads, head_dim).transpose(1, 2)
        # 缩放点积注意力计算
        Z = ScaledDotProductAttention(Q, K, V, mask)
        Z = Z.transpose(1, 2).reshape(B, L, self.dim)
        O = self.WO(Z)
        return O

bs = 4
heads = 8
seq_len = 6
dim = 512
MHA = MultiHeadAttention(dim, heads)
dummy_X = torch.randn(bs, seq_len, dim)
dummy_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bool)
out = MHA(dummy_X, dummy_mask)

print("输入形状:", tuple(dummy_X.shape))
print("输出形状:", tuple(out.shape))
