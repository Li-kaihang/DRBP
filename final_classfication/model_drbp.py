"""
DRBP 多标签联合学习模型
=======================
替代 model_baseline.py 的 LAMPPRoBaseline (三头塌缩: 两两相关 0.998~1.000)。

关键改动:
1. 三个标签各自独立的两层 MLP —— 旧版共用一个 nn.Linear(d,1), 结构上必然塌缩
2. 第三个头从 non 换成 DRBP, 用 y_dbp*y_rbp 直接监督。旧版 non = ¬dbp∧¬rbp
   完全由前两个决定, 不含独立信息
3. 删掉 LabelAwareAttention / CrossLabelAttention (3 个 token 上做自注意力 = 反复平均)
4. 加性残差 cross-talk (α 初始 0), 不是乘性门控 —— 训练集 DBP/RBP 负相关
   (lift=0.27), 乘性门控会被 5199:214 的样本比压成互相抑制
5. joint 模式修掉 CLS 错位 (上游 model.py:1573-1576 把两路截到 min_L 就 concat,
   ESM 第 0 位是 CLS, 导致整条序列相对结构错位一格)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_baseline import GatedResidual, MultiHeadAttention   # 这两个类本身是对的


class SharedTrunk(nn.Module):
    """
    共享 trunk: 多尺度 Conv1d + MHSA + 门控残差。

    产出残基级特征 (B,L,d), 交给 LabelQueryPooling 做 per-label pooling。
    三个头共用 trunk 是因为 DNA/RNA 结合有共同物理化学基础 (正电荷、芳香堆叠、
    锌指/RRM 等折叠); 但"读完之后怎么浓缩"要各标签各读各的, 不能共享同一个
    pool_q 平均 (那会逼两个头互斥, 见 LabelQueryPooling 注释)。
    """

    def __init__(self, d_model, n_heads=8, dropout=0.3):
        super().__init__()
        self.c3 = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.c5 = nn.Conv1d(d_model, d_model, 5, padding=2)
        self.c7 = nn.Conv1d(d_model, d_model, 7, padding=3)
        self.bn = nn.BatchNorm1d(d_model)
        self.conv_norm = nn.LayerNorm(d_model)

        self.mhsa = MultiHeadAttention(d_model, n_heads, dropout)
        self.gate = GatedResidual()
        self.attn_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, residue_mask):
        """x: (B,L,d)  residue_mask: (B,L) 1=真实残基。返回残基级特征 (B,L,d)。"""
        xt = x.transpose(1, 2)                                   # (B,d,L)
        conv = F.relu(self.bn(self.c3(xt) + self.c5(xt) + self.c7(xt)))
        x = self.conv_norm(x + conv.transpose(1, 2))             # 残差

        attn = self.mhsa(x, x, key_mask=residue_mask)
        x = self.attn_norm(self.gate(x, attn))
        return self.drop(x)


class LabelQueryPooling(nn.Module):
    """
    每个标签一个可学习 query, 各自 attend 整条残基序列 → 三个不同的 pooled 向量。

    动机 (2026-08-28): 旧版 SharedTrunk 用一个 pool_q 把所有残基加权平均成一个向量,
    三个头共用它。训练集里绝大多数蛋白非 DBP 即 RBP (DRBP 只占 1.7%), 模型最省事
    的解就是让这个共享向量"要么偏 DNA 要么偏 RNA", 于是两个头学成互斥 —— 实测真 DRBP
    上两主头同时开火只有 21/103, corr_dr 全程为负 (EZL 上 -0.855)。

    这里让 DBP 头自己 attend 到 DNA 结合域、RBP 头自己 attend 到 RNA 结合域, 同一个
    蛋白可以读出两个不同的向量, 物理上让"同时开火"成为可能。三个 query 独立初始化、
    独立参数, 不会被同一个 pooling 拉回平均。分类头仍用三个独立 MLP (塌缩的解药,
    不动), 只换 pooling 层。
    """

    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        self.d = d_model
        self.q_d = nn.Parameter(self._init_q())
        self.q_r = nn.Parameter(self._init_q())
        self.q_b = nn.Parameter(self._init_q())
        self.k_d = nn.Linear(d_model, d_model)
        self.k_r = nn.Linear(d_model, d_model)
        self.k_b = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _init_q(self):
        return torch.randn(1, self.d) * (self.d ** -0.5)

    def forward(self, x, residue_mask):
        """x: (B,L,d)  residue_mask: (B,L)。返回三个 (B,d)。"""
        neg = residue_mask.unsqueeze(-1) < 0.5              # (B,L,1), True=padding
        out = []
        for q, k in ((self.q_d, self.k_d), (self.q_r, self.k_r), (self.q_b, self.k_b)):
            keys = k(x)                                      # (B,L,d)
            score = (keys * q).sum(-1, keepdim=True)         # (B,L,1)
            score = score.masked_fill(neg, -1e9)
            w = score.softmax(dim=1)
            out.append((x * w).sum(dim=1))                   # (B,d)
        return out                                          # [h_d, h_r, h_b]


class MaskedCrossLabelAttention(nn.Module):
    """
    标签间注意力, 带 mask。目标: 让 RBP 头的证据显式地帮 DBP 头开火 (反之亦然),
    解决"真 DRBP 被误判成 RBP-only"。

    与 LAMP-PRo 的 cross-label attention 方向一致, 但这里:
      1. 三个头仍是独立 MLP (不是共享头), 所以不会塌缩成 0.998 相关;
      2. 加 mask 控制谁看谁 —— DBP↔RBP 互看, DRBP 头可以看两个主头(DRBP=两者都要),
         但主头不看 DRBP 头 (那个头是 1.7% 稀有类, 校准坏, 不让它的噪声回流)。

    mask: rows=query(要更新的标签), cols=key(被看的标签)。1=可看, 0=不可看。
      M = [[1,1,0],   # DBP 看 DBP 自己 + RBP
           [1,1,0],   # RBP 看 RBP 自己 + DBP
           [1,1,1]]   # DRBP 看所有 (含自己, 方便保留自身证据)
    """
    def __init__(self, d_model, n_heads=4, dropout=0.3):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.gate = GatedResidual()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        # 布尔 mask: False 的位置用 -1e9 掩掉 (软掉 softmax)。只让 DBP↔RBP 互看。
        self.register_buffer('mask', torch.tensor(
            [[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool), persistent=False)

    def forward(self, c):
        """c: (B, 3, d) 三个标签表征。返回 (B, 3, d)。"""
        B = c.shape[0]
        Q = self.W_q(c).view(B, 3, self.n_heads, self.d_k).transpose(1, 2)  # (B,H,3,dk)
        K = self.W_k(c).view(B, 3, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(c).view(B, 3, self.n_heads, self.d_k).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)     # (B,H,3,3)
        mask = self.mask.view(1, 1, 3, 3).expand(B, self.n_heads, -1, -1)
        attn = attn.masked_fill(~mask, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, 3, -1)
        out = self.W_o(out)
        return self.norm(self.gate(c, out))


class DRBPNetNew(nn.Module):
    """
    mode='esm2'  : 只用 ESM-2
    mode='joint' : ESM-2 + 结构 GNN, 残基级对齐后 concat
    mode='struct': 只用结构 GNN (纯结构消融, 无序列)
    """

    def __init__(self, config, esm_model=None, gnn_model=None):
        super().__init__()
        self.config = config
        self.mode = getattr(config, 'mode', 'esm2')
        d = config.d_model
        h = getattr(config, 'head_hidden', 64)

        self.esm = esm_model
        if esm_model is not None and config.freeze_esm:
            for p in self.esm.parameters():
                p.requires_grad = False

        # 纯结构模式不走序列分支, 不创建 input_proj (否则 DDP find_unused_parameters=False 报错)
        self.input_proj = (nn.Linear(config.esm_embed_dim, d)
                           if self.mode != 'struct' else None)

        self.gnn = gnn_model
        if self.mode == 'joint':
            gd = getattr(config, 'gnn_dim', 256)
            self.joint_proj = nn.Sequential(
                nn.Linear(d + gd, d), nn.LayerNorm(d), nn.ReLU())
        elif self.mode == 'struct':
            gd = getattr(config, 'gnn_dim', 256)
            self.struct_proj = nn.Sequential(
                nn.Linear(gd, d), nn.LayerNorm(d), nn.ReLU())

        self.trunk = SharedTrunk(d, config.n_heads, config.dropout)
        self.label_pool = LabelQueryPooling(d, config.dropout)
        # non 方案用共激活门控 f 代替 cross-label attention, 不创建这个模块 (否则 DDP 报未使用参数)
        self.cross_label = (MaskedCrossLabelAttention(d, getattr(config, 'cla_heads', 4),
                                                      config.dropout)
                            if getattr(config, 'head_scheme', 'drbp') == 'drbp' else None)

        # 三个独立的两层 MLP —— 拆成 fc1/fc2 是为了在中间插 cross-talk
        def fc1():
            return nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(config.dropout))

        self.head_scheme = getattr(config, 'head_scheme', 'drbp')   # 'drbp' | 'non'
        self.use_gate = getattr(config, 'use_gate', True)           # 共激活门控开关 (消融用)
        self.dbp_fc1, self.rbp_fc1 = fc1(), fc1()
        self.dbp_fc2 = nn.Linear(h, 1)
        self.rbp_fc2 = nn.Linear(h, 1)

        if self.head_scheme == 'non':
            # non 头 (非结合) —— 三头全是常见类 (DBP/RBP/non), 无稀有类
            self.non_fc1 = fc1()
            self.non_fc2 = nn.Linear(h, 1)
            # 共激活门控: f_d(h_r) = "DBP 从 RBP 提取多少信息", f_r(h_d) 反之。
            # 注意 f 可正可负(自己学), 配合过采样让它学会"双结合→顶, 纯RBP→压"。
            # use_gate=False 时不创建, 避免 DDP 报 unused parameters。
            if self.use_gate:
                self.f_d = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 1))
                self.f_r = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 1))
        else:
            self.drbp_fc1 = fc1()
            self.drbp_fc2 = nn.Linear(h, 1)

        # 加性残差 cross-talk。必须放在各头 fc1 之后 —— trunk 输出是共享的,
        # 在那一层做 cross-talk 等于 h + α·f(h), 自己跟自己交互, 没有意义。
        # non 方案用共激活门控代替 cross-talk/cross-label, 不再用这两个。
        self.use_crosstalk = getattr(config, 'use_crosstalk', True) and self.head_scheme == 'drbp'
        self.use_cross_label = getattr(config, 'use_cross_label', True) and self.head_scheme == 'drbp'
        if self.use_crosstalk:
            self.f_r2d = nn.Linear(h, h)
            self.f_d2r = nn.Linear(h, h)
            # α 初始化为 0 → 起点是恒等映射, 模型有权选择不交互。
            # 训练完看 α 学成多少, 本身就是"DBP/RBP 该不该互相看"的实验结论。
            self.alpha_d = nn.Parameter(torch.zeros(1))
            self.alpha_r = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    def encode(self, input_ids, attention_mask, residue_mask,
               struct_batch=None, res2struct=None):
        """返回残基级表征 (B, Lr, d)"""
        Lr = residue_mask.shape[1]

        if self.mode == 'struct':
            # 纯结构: 只用 GNN。空结构 (mask 全 0) → struct_proj(0) = 固定偏置,
            # 即"无结构蛋白只能按先验猜", 与 joint 的"空结构退化成纯序列"哲学一致。
            g = self.gnn(struct_batch)                           # (B, Ls, gd)
            g = g * struct_batch['mask'].unsqueeze(-1)
            valid = (res2struct >= 0).unsqueeze(-1)              # (B, Lr, 1)
            idx = res2struct.clamp(min=0).unsqueeze(-1).expand(-1, -1, g.shape[-1])
            g_al = g.gather(1, idx) * valid                      # (B, Lr, gd)
            return self.struct_proj(g_al)                        # (B, Lr, d)

        out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state                              # (B, Lt, 640)
        assert seq.shape[1] >= Lr + 1, (
            f"ESM 输出长度 {seq.shape[1]} 不足以取 {Lr} 个残基 (需 >= Lr+1)")
        seq = seq[:, 1:1 + Lr]        # 丢掉 CLS, 取残基 0..Lr-1  ← 修 CLS 错位
        h = self.input_proj(seq)                                 # (B, Lr, d)

        if self.mode == 'joint':
            g = self.gnn(struct_batch)                           # (B, Ls, gd)
            # 先按结构 mask 清零 —— 空结构 (mask 全 0) 的贡献必须精确为 0
            g = g * struct_batch['mask'].unsqueeze(-1)

            # 按 res2struct 把结构特征 gather 到全序列的残基位置上
            valid = (res2struct >= 0).unsqueeze(-1)              # (B, Lr, 1)
            idx = res2struct.clamp(min=0).unsqueeze(-1).expand(-1, -1, g.shape[-1])
            g_al = g.gather(1, idx) * valid                      # (B, Lr, gd)

            h = self.joint_proj(torch.cat([h, g_al], dim=-1))
        return h

    def forward(self, input_ids, attention_mask, residue_mask,
                struct_batch=None, res2struct=None):
        h = self.encode(input_ids, attention_mask, residue_mask,
                        struct_batch, res2struct)
        # label-aware pooling: 三个头各自 attend 残基序列, 产出三个不同的向量
        feat = self.trunk(h, residue_mask)                       # (B,L,d)
        h_d, h_r, h_b = self.label_pool(feat, residue_mask)      # 三个 (B,d)

        if self.head_scheme == 'non':
            # non 方案: 第三 query 是 non(非结合), 不是 DRBP
            h_n = h_b
            z_d = self.dbp_fc2(self.dbp_fc1(h_d)).squeeze(-1)
            z_r = self.rbp_fc2(self.rbp_fc1(h_r)).squeeze(-1)
            # 共激活门控: 每个头先独立打分, 再根据对方信息微调
            if self.use_gate:
                z_d = z_d + self.f_d(h_r).squeeze(-1)            # DBP 从 RBP 提取信息
                z_r = z_r + self.f_r(h_d).squeeze(-1)            # RBP 从 DBP 提取信息
            z_n = self.non_fc2(self.non_fc1(h_n)).squeeze(-1)
            dbp_prob = torch.sigmoid(z_d)
            rbp_prob = torch.sigmoid(z_r)
            return {
                'z_dbp': z_d, 'z_rbp': z_r, 'z_non': z_n,
                'dbp_prob': dbp_prob, 'rbp_prob': rbp_prob,
                'non_prob': torch.sigmoid(z_n),
                'drbp_prob': dbp_prob * rbp_prob,                # 派生(软AND), 供评估
            }

        # cross-label attention: 让 RBP 的证据帮 DBP 开火 (反之亦然), 解决"真 DRBP
        # 被误判成 RBP-only"。只在 label-aware pooling 之后、进入独立 MLP 之前做,
        # 三个 MLP 头保持独立 (这是之前 LAMPPRoBaseline 塌缩的解药, 不能丢)。
        if self.use_cross_label:
            c = torch.stack([h_d, h_r, h_b], dim=1)              # (B,3,d)
            c = self.cross_label(c)
            h_d, h_r, h_b = c[:, 0], c[:, 1], c[:, 2]

        h_d = self.dbp_fc1(h_d)
        h_r = self.rbp_fc1(h_r)
        h_b = self.drbp_fc1(h_b)

        if self.use_crosstalk:
            # 用交叉前的 h_d/h_r 互相喂, 避免一路被更新后再影响另一路
            d_in, r_in = h_d, h_r
            h_d = d_in + self.alpha_d * self.f_r2d(r_in)
            h_r = r_in + self.alpha_r * self.f_d2r(d_in)

        z_d = self.dbp_fc2(h_d).squeeze(-1)
        z_r = self.rbp_fc2(h_r).squeeze(-1)
        z_b = self.drbp_fc2(h_b).squeeze(-1)

        return {
            'z_dbp': z_d, 'z_rbp': z_r, 'z_drbp': z_b,
            'dbp_prob': torch.sigmoid(z_d),
            'rbp_prob': torch.sigmoid(z_r),
            'drbp_prob': torch.sigmoid(z_b),
        }

    def alphas(self):
        if not self.use_crosstalk:
            return 0.0, 0.0
        return float(self.alpha_d.detach()), float(self.alpha_r.detach())


# ======================================================================
# 联合损失
# ======================================================================

def drbp_loss(out, y_d, y_r, w_d, w_r, w_b, lam1=1.0, lam2=0.3, cons_mode='teacher',
              w_cons_pos=0.0, lam3=0.0, conj_margin=1.0):
    """
    L = BCE(z_d,y_d,w_d) + BCE(z_r,y_r,w_r)
      + λ1·BCE(z_b, y_d·y_r, w_b)          ← DRBP 直接监督
      + λ2·cons                             ← 层级一致性 ("传送带")
      + λ3·conj                             ← 共现边际 (新)

    第三项是核心修复: 旧版 DRBP 从未被直接监督, 只是两个阈值的事后交集, 而边际 BCE
    会持续把双阳往下压 (训练集 lift=0.245)。

    第四项把 "DRBP = DBP ∧ RBP" 的逻辑约束显式写进 loss。最终 4 类判定用的是
    dbp_p/rbp_p 两个主头, DRBP 头**不参与判定**, 所以 DRBP 头学到的东西要传到主头
    上, 只能靠共享 trunk (间接) 和这一项 (直接)。λ2 才是"传送带"。

    ---- w_cons_pos: 修传送带反转 (2026-08-27) ----
    原来 cons 是 F.mse_loss(...) 即**全 batch 平均**, 没有任何加权。但 DRBP 只占
    1.7%: 有效 batch 32 条 → 每个优化步平均只有 0.55 个 DRBP, 约 58% 的步里一个都
    没有。于是这一项在 98.3% 的样本上说"把 σ(z_d)·σ(z_r) 压低", 只在 1.7% 上说
    "抬高" —— 平均下来它在教两个主头**互斥**, 正好和意图相反。
    l_d/l_r/l_b 三项都有 pos_weight 补偿, 只有 cons 裸奔。

    实测后果 (DRBP206 的 103 个真 DRBP, ep3 模型):
        DBP 头开火 42/103 (中位概率 0.382), RBP 头开火 74/103 (中位 0.878)
        两主头同时开火只有 21/103  ←  σ(z_d)·σ(z_r)=0.335 而 σ(z_b)=0.633
    传送带本该把乘积抬到 0.633, 实际被压在 0.335。

    w_cons_pos 在全 batch 平均之外**额外**加一份只在真 DRBP 上的误差。
    w_cons_pos=0 → 完全等价于旧行为 (消融对照组用这个)。

    ---- λ3 / conj_margin: 共现边际损失 (2026-08-27) ----
    直接盯住"较弱的那个头": 在真 DRBP 样本上惩罚 min(z_d, z_r) 低于 margin 的部分。
    实测弱头是 DBP。用 logit 空间的 softplus 而非概率空间, 梯度在饱和区更稳。
    conj_margin=1.0 对应 σ(1.0)≈0.73, 即要求两个头都至少到 0.73。
    λ3=0 → 关闭 (消融对照组用这个)。

    注意方向: LAMP-PRo 的 InvalidLabelPenalty 管的是"**不能**同时为真"的非法组合;
    这一项管的是"**应该**同时为真"。两者相反, 不是同一个东西。

    cons_mode:
      'teacher' (默认) —— detach 掉 DRBP 头那一侧, 单向传递。
      'sym'            —— 对称 MSE。缺点: DRBP 头确信而主头不确信时, 在推主头上去的
                          同时也把 DRBP 头往下拽。

    用 MSE 而非 BCE: 目标 σ(z_d)·σ(z_r) 是软目标, BCE 在 0/1 附近梯度不稳。
    """
    y_b = y_d * y_r
    l_d = F.binary_cross_entropy_with_logits(out['z_dbp'], y_d, pos_weight=w_d)
    l_r = F.binary_cross_entropy_with_logits(out['z_rbp'], y_r, pos_weight=w_r)
    l_b = F.binary_cross_entropy_with_logits(out['z_drbp'], y_b, pos_weight=w_b)

    target = out['drbp_prob'].detach() if cons_mode == 'teacher' else out['drbp_prob']
    err = (out['dbp_prob'] * out['rbp_prob'] - target) ** 2
    cons = err.mean()                                   # 旧行为, 保留
    pos = (y_b > 0.5)
    n_pos = int(pos.sum())
    if w_cons_pos > 0 and n_pos > 0:
        # 额外补一份只在真 DRBP 上的误差 —— 抵消 98.3% 负样本对这一项的主导
        cons = cons + w_cons_pos * err[pos].mean()

    # 共现边际: 只作用在真 DRBP 上, 不直接推非 DRBP 样本 (那会造假阳)
    if lam3 > 0 and n_pos > 0:
        z_min = torch.minimum(out['z_dbp'], out['z_rbp'])
        conj = F.softplus(conj_margin - z_min[pos]).mean()
    else:
        conj = torch.zeros((), device=out['z_dbp'].device)

    total = l_d + l_r + lam1 * l_b + lam2 * cons + lam3 * conj
    return total, {'l_dbp': l_d.item(), 'l_rbp': l_r.item(),
                   'l_drbp': l_b.item(), 'l_cons': cons.item(),
                   'l_conj': float(conj)}


def non_loss(out, y_d, y_r, w_d, w_r, w_n):
    """
    non 方案损失 (三头: DBP / RBP / non)。DRBP 不单独监督, 由 DBP∧RBP 派生。

    L = BCE(z_d, y_d, w_d) + BCE(z_r, y_r, w_r) + BCE(z_n, y_n, w_n)

    关键: 三头全是常见类 (DBP~34% / RBP~21% / non~47%), 没有稀有类,
    pos_weight 都在 1~3 之间, 不再需要 w_b=40 那种极端权重 → 校准天然正常。

    共激活门控 f 不单独加 loss, 它的梯度来自 l_d/l_r: 真 DRBP 上 y_d=1、y_r=1,
    两个头都要被推高, f 自然会学会"在对方信息暗示双结合时给正 boost"。
    配合过采样让 DRBP 信号足够强, 防止 f 学成"一直负"(互斥)。
    """
    y_n = ((y_d == 0) & (y_r == 0)).float()
    l_d = F.binary_cross_entropy_with_logits(out['z_dbp'], y_d, pos_weight=w_d)
    l_r = F.binary_cross_entropy_with_logits(out['z_rbp'], y_r, pos_weight=w_r)
    l_n = F.binary_cross_entropy_with_logits(out['z_non'], y_n, pos_weight=w_n)
    total = l_d + l_r + l_n
    return total, {'l_dbp': l_d.item(), 'l_rbp': l_r.item(), 'l_non': l_n.item()}
