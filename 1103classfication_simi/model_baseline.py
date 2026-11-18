#!/usr/bin/env python3
"""
LAMP-PRo 风格序列 baseline (DRBP 分类)
========================================
架构 (参考 LAMP-PRo, arXiv:2509.24262):

    ESM-2(150M, 640) ──> Linear(640→d_model)
        └─> 1D CNN (局部)
        └─> MHSA (全局, 门控残差)
        └─> 标签感知注意力 (label-aware attention)
              Q = 可学习标签向量 (C 个), K/V = 蛋白序列特征
              → 每标签一个表征 c_c
        └─> 跨标签注意力 (cross-label attention, 门控残差)
              让 C 个标签表征互相 attend → 建模 DBP↔RBP 共现 (DRBP)
        └─> Linear + sigmoid → C 个概率

标签方案: 3 类 multi-label (对齐 LAMP-PRo)
    DBP       = [1, 0, 0]
    RBP       = [0, 1, 0]
    non-NABP  = [0, 0, 1]  (双负, 由 DBP=0 & RBP=0 派生)
    DRBP      = [1, 1, 0]  (双正, 不单独设类)

注意: 这是对 LAMP-PRo 的高层重建 (原文代码当时拉不到),
      Q/K/V 取向按论文描述 + 标准 label-attention 文献实现, 可后续逐行对齐。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ============================================================
# 配置
# ============================================================

class Config:
    def __init__(self, **kw):
        # ESM-2
        self.esm_model_name = "facebook/esm2_t30_150M_UR50D"
        self.esm_embed_dim = 640
        self.freeze_esm = True            # 默认冻结 ESM, 只训轻量头 (提泛化)
        # 主干
        self.d_model = 256
        self.n_heads = 8
        self.dropout = 0.3
        # 标签
        self.n_labels = 3                 # DBP / RBP / non-NABP
        # 训练
        self.batch_size = 4
        self.learning_rate = 1e-3
        self.weight_decay = 5e-4
        self.num_epochs = 30
        self.max_seq_len = 1024
        self.label_smoothing = 0.0
        # 覆盖
        for k, v in kw.items():
            setattr(self, k, v)


# ============================================================
# 门控残差
# ============================================================

class GatedResidual(nn.Module):
    """
    x' = x + sigmoid(g) * F(x)
    g 初始为负 → 初始接近 identity, 训练更稳 (LAMP-PRo 的门控残差思想)。
    """

    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        return x + torch.sigmoid(self.gate) * fx


# ============================================================
# 多头注意力 (支持 cross-attention)
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    通用多头注意力。Q 来自 q, K/V 来自 kv。
    若 q is kv 则为自注意力。
    key_mask: (B, L_kv) 1=有效位置。
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Lq, _ = q.shape
        Bk, Lkv, _ = kv.shape

        Q = self.W_q(q).view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(kv).view(Bk, Lkv, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(kv).view(Bk, Lkv, self.n_heads, self.d_k).transpose(1, 2)

        scale = math.sqrt(self.d_k)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, Lq, Lkv)

        if key_mask is not None:
            # key_mask: (B, Lkv) → (B, 1, 1, Lkv)
            m = key_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(m < 0.5, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, H, Lq, d_k)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.W_o(out)


# ============================================================
# 标签感知注意力
# ============================================================

class LabelAwareAttention(nn.Module):
    """
    C 个可学习标签向量作为 query, attend 蛋白序列特征 H (B, L, d) → 每标签表征 (B, C, d)。

    每个标签 c 得到整条序列里"与它最相关"的浓缩摘要。
    取代原来的 ResidueLevelDisentangler 两层 MLP (盲投影)。
    """

    def __init__(self, d_model: int, n_labels: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_labels = n_labels
        self.label_emb = nn.Parameter(torch.randn(n_labels, d_model) * 0.02)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = H.shape[0]
        # Q: 标签向量扩展到 batch
        Q = self.label_emb.unsqueeze(0).expand(B, self.n_labels, -1)  # (B, C, d)
        c = self.attn(Q, H, key_mask=mask)  # (B, C, d)
        return self.norm(Q + c)


# ============================================================
# 跨标签注意力
# ============================================================

class CrossLabelAttention(nn.Module):
    """
    让 C 个标签表征互相 attend (自注意力), 建模标签间依赖 (DBP↔RBP → DRBP)。
    门控残差连接。
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.gate = GatedResidual()

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(c, c)  # 自注意力 over labels
        return self.norm(self.gate(c, attn_out))


# ============================================================
# 主模型
# ============================================================

class LAMPPRoBaseline(nn.Module):
    """
    LAMP-PRo 风格序列 baseline。
    """

    def __init__(self, config: Config, esm_model=None):
        super().__init__()
        self.config = config
        d = config.d_model

        self.esm = esm_model
        # 冻结 ESM
        if esm_model is not None and config.freeze_esm:
            for p in esm_model.parameters():
                p.requires_grad = False

        # 投影 640 → d_model
        self.input_proj = nn.Linear(config.esm_embed_dim, d)

        # 1D CNN (局部特征)
        self.conv = nn.Sequential(
            nn.Conv1d(d, d, kernel_size=3, padding=1),
            nn.BatchNorm1d(d),
            nn.ReLU(),
        )

        # MHSA (全局, 门控残差)
        self.mhsa = MultiHeadAttention(d, config.n_heads, config.dropout)
        self.mhsa_norm = nn.LayerNorm(d)
        self.mhsa_gate = GatedResidual()

        # 标签感知注意力
        self.label_attn = LabelAwareAttention(d, config.n_labels, config.n_heads, config.dropout)

        # 跨标签注意力
        self.cross_label = CrossLabelAttention(d, config.n_heads, config.dropout)

        # 分类头: 每标签向量 → 1 个 logit
        self.head = nn.Linear(d, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for m in self.conv.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Args:
            input_ids:      (B, L)
            attention_mask: (B, L)  1=有效位置
        Returns:
            dict:
              logits   (B, 3)   DBP/RBP/non-NABP 的原始 logits
              dbp_prob (B, 1)   sigmoid
              rbp_prob (B, 1)
              non_prob (B, 1)
        """
        # ESM-2
        if self.esm is not None:
            out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
            H = out.last_hidden_state  # (B, L, 640)
        else:
            # 无 ESM (测试结构时), 假设输入已是 embedding
            H = input_ids

        B, L, _ = H.shape

        # 投影
        H = self.input_proj(H)  # (B, L, d)

        # CNN: (B, L, d) → (B, d, L) → conv → (B, L, d)
        H = H.transpose(1, 2)
        H = self.conv(H)
        H = H.transpose(1, 2)

        # MHSA + 门控残差
        attn_out = self.mhsa(H, H, key_mask=attention_mask)
        H = self.mhsa_norm(self.mhsa_gate(H, attn_out))

        # 标签感知注意力 → (B, C, d)
        c = self.label_attn(H, mask=attention_mask)

        # 跨标签注意力 → (B, C, d)
        c = self.cross_label(c)

        # 头: 每标签一个 logit → (B, C)
        logits = self.head(c).squeeze(-1)  # (B, 3)

        dbp_prob = torch.sigmoid(logits[:, 0:1])
        rbp_prob = torch.sigmoid(logits[:, 1:2])
        non_prob = torch.sigmoid(logits[:, 2:3])

        return {
            'logits': logits,
            'dbp_prob': dbp_prob,
            'rbp_prob': rbp_prob,
            'non_prob': non_prob,
        }
