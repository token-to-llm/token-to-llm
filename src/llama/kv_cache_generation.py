"""带 KV-Cache 的生成过程（教材 4.3.2 节 · 代码清单「带KV-Cache的生成过程」）。"""

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


class MultiHeadAttentionWithKVCache(nn.Module):
    def __init__(self, dim, heads):
        super(MultiHeadAttentionWithKVCache, self).__init__()
        self.heads = heads
        self.dim = dim
        self.WQ = nn.Linear(dim, dim)
        self.WK = nn.Linear(dim, dim)
        self.WV = nn.Linear(dim, dim)
        self.WO = nn.Linear(dim, dim)
        self.cache_K = None
        self.cache_V = None

    def forward(self, X, mask):
        B, L, D = X.size()
        head_dim = self.dim // self.heads  # 单头词向量维度
        Q, K, V = self.WQ(X), self.WK(X), self.WV(X)
        Q = Q.reshape(B, L, self.heads, head_dim).transpose(1, 2)
        K = K.reshape(B, L, self.heads, head_dim).transpose(1, 2)
        V = V.reshape(B, L, self.heads, head_dim).transpose(1, 2)

        if self.cache_K is None:
            self.cache_K = K
            self.cache_V = V
        else:
            self.cache_K = torch.cat((self.cache_K, K), dim=2)
            self.cache_V = torch.cat((self.cache_V, V), dim=2)

        Z = ScaledDotProductAttention(Q,
                                      self.cache_K, self.cache_V,
                                      mask)

        Z = Z.transpose(1, 2).reshape(B, L, self.dim)
        O = self.WO(Z)
        return O


class DecodingModelWithKVCache(nn.Module):
    def __init__(self, dim, heads, vocab_size):
        super(DecodingModelWithKVCache, self).__init__()
        self.embd = nn.Embedding(vocab_size, dim)  # 忽略位置编码
        self.decoder = MultiHeadAttentionWithKVCache(dim, heads)  # 简化解码器
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

seq_len = 16
x = torch.randint(low=1, high=vocab_size, size=(1, seq_len))
mask = torch.triu(torch.ones(1, 1,
                             seq_len+max_new_tokens,
                             seq_len+max_new_tokens,
                             dtype=torch.bool))[:, :, 1:, :-1]
pos = torch.arange(seq_len)

# 带 KV-Cache 的模型
model = DecodingModelWithKVCache(dim, heads, vocab_size)
for i in range(max_new_tokens):
    logits = model(x, mask[:, :, pos, pos])
    next_tokens = torch.argmax(logits, dim=-1)

    # x = torch.cat((x, next_tokens[:, [-1]]), dim = -1)
    x = next_tokens[:, [-1]]  # 有 KV-Cache 更新情况

    pos = torch.tensor([seq_len+i])
    if next_tokens[0, -1] == EOS_token_ID:
        break

print("输入形状:", (1, seq_len))
print("logits 形状:", tuple(logits.shape))
print("输出形状:", tuple(x.shape))
print("KV-Cache 形状:", tuple(model.decoder.cache_K.shape))
