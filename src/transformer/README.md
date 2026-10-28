# 第 2 章　Transformer 模型

## 代码清单

| 脚本 | 教材清单 | 对应 |
|---|---|---|
| `scaled_dot_product_attention.py` | 代码 1　缩放点积注意力 Pytorch 代码 | 2.3.1 节 |
| `multi_head_attention.py` | 代码 2　多头注意力机制 Pytorch 代码 | 2.3.1 节 |

> 教材清单里的示例掩码写作 `torch.ones(...)`，按该函数的掩码语义（`True` 处屏蔽）
> 会使输出全为 NaN，这里改为 `torch.zeros(...)`。
> 另见仓库根目录 `error.md` 的〈代码问题〉C.1。

本章无绘图脚本。
