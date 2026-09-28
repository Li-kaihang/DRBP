#!/usr/bin/env python3
"""联合 binding site 模型: 原始 ESM-2 (冻结) + StructureGNN (含表面特征) → cat → 位点头。

步骤②(+结构): 在纯序列 ESM-2 基础上加结构 GNN, 用同一原始 ESM-2 重训, 便于消融对比。
"""
import torch
import torch.nn as nn


class JointBindingSiteModel(nn.Module):
    def __init__(self, esm, struct_gnn, d_seq=640, d_struct=256, hidden=256,
                 seq_only_dna=False, dropout=0.0, context_head=False, context_layers=2,
                 gated_fusion=False):
        super().__init__()
        self.esm = esm                  # 冻结
        self.struct_gnn = struct_gnn    # 可训练
        self.seq_only_dna = seq_only_dna
        self.context_head = context_head
        self.gated_fusion = gated_fusion
        if gated_fusion:
            # 门控融合: 序列/结构各自投影到 hidden, 每个头独立学 gate(0~1) 决定信结构多少。
            # 治「concat 把 256 维结构稀释进 1536 维」: 结构作为可开关的正交证据参与, 而非混进 concat。
            self.seq_proj = nn.Linear(d_seq, hidden)
            self.struct_proj = nn.Linear(d_struct, hidden)
            self.dna_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
            self.rna_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        else:
            self.fuse = nn.Sequential(nn.Linear(d_seq + d_struct, hidden), nn.ReLU(),
                                      nn.Dropout(dropout))
        # 残基间上下文编码: 结合位点是空间/序列上的连续斑块, 逐残基独立 MLP 头会丢
        # 掉这种"邻域一致性", 产生孤立假阳性。加 2 层 self-attention 让每个残基聚合
        # 窗口内其它残基的融合特征, 抑制孤立误报、提升精度。
        if context_head:
            layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=8,
                                               dim_feedforward=hidden * 4,
                                               dropout=dropout, activation='gelu',
                                               batch_first=True, norm_first=True)
            self.context = nn.TransformerEncoder(layer, num_layers=context_layers)
        # DNA 头只吃序列时: 序列(640) 先投到 hidden, 绕开结构特征
        if seq_only_dna:
            self.dna_proj = nn.Sequential(nn.Linear(d_seq, hidden), nn.ReLU(),
                                          nn.Dropout(dropout))
        self.dna_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(64, 1))
        self.rna_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, struct_batch, input_ids, attention_mask):
        # 序列 ESM-2 → (B, L+2, 640), 去掉 CLS/EOS
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state[:, 1:-1, :]            # (B, L_seq, 640)

        # 结构 GNN → (B, L_struct, 256)
        struct = self.struct_gnn(struct_batch)

        # 对齐长度 (取两者较短)
        L = min(seq.shape[1], struct.shape[1])
        seq = seq[:, :L, :]
        struct = struct[:, :L, :]

        if self.gated_fusion:
            seq_p = self.seq_proj(seq)                          # (B, L, hidden)
            struct_p = self.struct_proj(struct)                 # (B, L, hidden)
            cat = torch.cat([seq_p, struct_p], dim=-1)          # (B, L, 2*hidden)
            dna_gate = self.dna_gate(cat)                       # (B, L, hidden) ∈ [0,1]
            rna_gate = self.rna_gate(cat)
            dna_fused = dna_gate * struct_p + (1 - dna_gate) * seq_p
            rna_fused = rna_gate * struct_p + (1 - rna_gate) * seq_p
            if self.context_head:
                dna_fused = self.context(dna_fused)             # 残基间 self-attention（共享权重）
                rna_fused = self.context(rna_fused)
            dna_input = self.dna_proj(seq) if self.seq_only_dna else dna_fused
            dna_logits = self.dna_head(dna_input).squeeze(-1)   # (B, L)
            rna_logits = self.rna_head(rna_fused).squeeze(-1)   # (B, L)
        else:
            fused = self.fuse(torch.cat([seq, struct], dim=-1))   # (B, L, hidden)
            if self.context_head:
                fused = self.context(fused)                       # 残基间 self-attention
            dna_input = self.dna_proj(seq) if self.seq_only_dna else fused
            dna_logits = self.dna_head(dna_input).squeeze(-1)     # (B, L)
            rna_logits = self.rna_head(fused).squeeze(-1)         # (B, L)
        return dna_logits, rna_logits
