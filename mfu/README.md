
# attention 运算量计算

```python
'''
每个矩阵输出原始要考虑乘加的数量, (*) 里表示运行的次数
self_atten_block = 8 * B * L * h * h(Wq, Wk, Wv, Wo) + 4 * B * L * h * L(QK^T, PV^T)
cross_attn_block = 4 * B * L * h * h(Wq, Wo) + 4 * B * S * h * h(Wk, Wv) + 4 * B * L * S * h(QK^T, PV^T)
ffn_layer = 4 * B * L * ffn_dim * h
'''
# B = 4
# L = 1024
B = 8
L = 512
S = 512

B * L * h * h

def wan_flops(B, L, S, h=5120, ffn_dim=13824, num_layers=40, factor=3):
    flops_per_layer = 12 * B * L * h * h + 4 * B * S * h * h + 4 * B * L * L * h + 4 * B * L * S * h + 4 * B * L * ffn_dim * h
    return factor * num_layers * flops_per_layer
```

# 频率查询

```sh
nvidia-smi -i 0 -q |grep -E "(Graphics)|(Temp)|(Power)"
```

# 算力计算公式

```sh
clock_rate = 1950 MHZ
4096 * clock_rate * mp_count / 1000 / 1000 = 480 T
```