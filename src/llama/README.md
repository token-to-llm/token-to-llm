# 第 4 章　Llama 模型

## 代码清单

| 脚本 | 教材清单 | 对应 |
|---|---|---|
| `kv_cache_generation.py` | 代码 1　带KV-Cache的生成过程 | 4.3.2 节 |
| `kv_cache_attention.py` | 代码 2　带KV-Cache的注意力层 | 4.3.2 节 |

> 教材清单里掩码用 `mask[:, :, pos, pos]` 取子块，两个高级索引形状相同会成对配对，
> 取到的是主对角线而非子块，因果掩码实际未生效；这里改为逐位置对齐缓存长度的二维掩码。
> 另见仓库根目录 `error.md` 的〈代码问题〉C.3。

## 绘图

| 脚本 | 输出 | 对应 |
|---|---|---|
| `rmsnorm.py` | `output/rmsnorm.pdf` | 图 4.2　RMS归一化缩放效应图 |
| `sine_frequency.py` | `output/sine_frequency.pdf` | 图 4.14　正弦函数周期与频率的关系 |
| `position_encoding_vector.py` | `output/position_encoding_vector.pdf` | 图 4.15　位置编码向量随维度的变化图 |
| `rope_low_high.py` | `output/rope_low_high.pdf` | 图 4.16　高低频位置编码的点积衰减特性分析 |
| `position_interpolation.py` | `output/position_interpolation.pdf` | 图 4.17　位置编码内插前后对比 |
| `pi_rope.py` | `output/pi_rope.pdf` | 图 4.18　标准RoPE与PI-RoPE的注意力分数随相对位置衰减曲线对比图 |
| `ntk_rope.py` | `output/ntk_rope.pdf` | 图 4.19　放大基数对位置编码向量影响 |
| `three_rope_decay.py` | `output/three_rope_decay.pdf` | 图 4.20　三种位置编码的衰减性可视化 |
| `three_rope_low_high.py` | `output/three_rope_low_high.pdf` | 图 4.21　高低频位置编码的衰减性分析 |
| `rope_decay_length.py` | `output/rope_decay_length.pdf` | 图 4.12　相对位置编码衰减性示意图 |
| `rope_decay.py` | `output/rope_decay.pdf` | 图 4.13　RoPE 位置编码点积衰减曲线图 |
