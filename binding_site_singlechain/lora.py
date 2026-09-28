#!/usr/bin/env python3
"""LoRA (低秩适配) for ESM-2 的 q/k/v 投影。

冻结 ESM 全部权重, 只对后 n_layers 层的 query/key/value 加低秩增量 BA。
目标: 1062 条小数据上 unfreeze-6 的 118M 全秩微调严重过拟合 (val↗ test↘),
改用 LoRA 把可训练参数降到 ~几 M, 既保留 650M 预训练表征又避免过拟合。
"""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """包装现有 nn.Linear: 冻结原权重, 加低秩增量 (scaling * x @ A^T @ B^T)。

    A: (r, in), B: (out, r)。B 初始化为 0, 使初始 LoRA 增量 = 0 (从预训练权重起步)。
    """
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout)
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        delta = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * delta


def apply_lora_to_esm(esm, r=16, alpha=32, dropout=0.1, n_layers=6):
    """把 ESM 后 n_layers 层的 attention.self.query/key/value 替换成 LoRALinear。

    其余权重保持冻结 (调用方需先冻结全部 ESM 参数)。返回 esm 便于链式。
    """
    n = len(esm.encoder.layer)
    start = n - n_layers
    for i in range(start, n):
        self_attn = esm.encoder.layer[i].attention.self
        self_attn.query = LoRALinear(self_attn.query, r=r, alpha=alpha, dropout=dropout)
        self_attn.key = LoRALinear(self_attn.key, r=r, alpha=alpha, dropout=dropout)
        self_attn.value = LoRALinear(self_attn.value, r=r, alpha=alpha, dropout=dropout)
    return esm
