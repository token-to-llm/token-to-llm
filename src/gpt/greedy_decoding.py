"""贪婪解码（教材 3.2.2 节 · 代码清单「贪婪解码」）。"""

import math

import torch
import torch.nn as nn

torch.manual_seed(42)


def ScaledDotProductAttention(Q, K, V, mask):
    B, L, H, D = K.size()
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


class DecodingModel(nn.Module):
    def __init__(self, dim, heads, vocab_size):
        super(DecodingModel, self).__init__()
        self.embd = nn.Embedding(vocab_size, dim)  # 忽略位置编码
        self.decoder = MultiHeadAttention(dim, heads)  # 简化解码器
        self.norm = nn.LayerNorm(dim)
        self.lm_heads = nn.Linear(dim, vocab_size)

    def forward(self, x, mask):
        X = self.embd(x)
        X = self.decoder(X, mask)
        H = self.norm(X)
        logits = self.lm_heads(H)
        return logits

vocab_size = 100
heads = 8
dim = 64
max_new_tokens = 10
EOS_token_ID = 0

# 1个输入, 序列长度为16
bs = 1
seq_len = 16
x = torch.randint(low=1, high=vocab_size, size=(bs, seq_len))
mask = torch.triu(torch.ones(1, 1,
                             seq_len+max_new_tokens,
                             seq_len+max_new_tokens,
                             dtype=torch.bool))[:, :, 1:, :-1]
model = DecodingModel(dim, heads, vocab_size)

for i in range(max_new_tokens):
    logits = model(x, mask[:, :, :seq_len+i, :seq_len+i])
    next_tokens = torch.argmax(logits, dim=-1)
    x = torch.cat((x, next_tokens[:, [-1]]), dim=-1)
    if next_tokens[0, -1] == EOS_token_ID:
        break

print("输入形状:", (bs, seq_len))
print("logits 形状:", tuple(logits.shape))
print("输出形状:", tuple(x.shape))
