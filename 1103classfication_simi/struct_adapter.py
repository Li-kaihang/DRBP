"""
结构适配层
==========
把 new/data/parsed/*.csv (列: protein_id, seq, DBP_label, RBP_label) 接到
final/classification 的结构特征管线上, 但不修改那边任何文件。

要解决四件事:

1. **抽链**。final/classification 的 parse_pdb_backbone 没有 chain 参数, PDB255
   的多链结构会被拼成一条 (实测 4BHXA 解析出 182 残基, CSV 里只有 94)。这里自己
   实现一个 chain-aware 的骨架解析。

2. **残基对齐**。ESM 按完整序列索引, GNN 按结构里实际存在的残基索引。AlphaFold
   两者一致 (实测 298/300), 但 RCSB 真实结构有缺失残基, 必须建映射。
   产出 res2struct: 长度 Lr, res2struct[j] = 全序列残基 j 对应的结构下标, 无则 -1。

3. **结构缺失降级**。覆盖率 94.8%, 测试集 EZL 只有 79.5%。不过滤、不报错, 产出
   res2struct 全 -1 的空结构。这样 esm2 和 joint 两种模式跑的是同一批样本, 消融
   才公平; joint 对缺失样本自动退化成纯序列。

4. **特征缓存**。每条 PDB 解析+特征提取约 170ms, 18331 条每 epoch 重算要 49 分钟。
   预先缓存成 .pt。
"""

import os
import re
import difflib
import numpy as np
import torch

AF_DIR = "/root/DRBP/data/output/structure"
PDB255_DIR = "/root/DRBP/data/output/structure_pdb255"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "structure_cache")

UNIPROT_RE = re.compile(r"[A-Z0-9]{6}|[A-Z0-9]{10}")

THREE2ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'SEC': 'U', 'PYL': 'O',
    'MSE': 'M',   # 硒代甲硫氨酸, RCSB 结构里常见
}

# 特征提取产出的、需要按残基维度 padding 的字段
STRUCT_KEYS = [
    'aa_indices', 'dihedral_sincos', 'backbone_frames_R', 'backbone_frames_t',
    'backbone_frames_quat', 'backbone_geom', 'local_atom_coords',
]


# ============================================================
# 1. chain-aware 骨架解析
# ============================================================

def parse_backbone(pdb_path, chain=None):
    """
    读 PDB 骨架原子。chain=None 时读第一条链 (AlphaFold 模型只有一条)。

    返回 {'N','CA','C','O': (L,3) float32, 'sequence': str, 'L': int}
    与 final/classification 的 parse_pdb_backbone 返回格式一致, 但支持选链。
    """
    residues = {}          # (chain, resseq, icode) -> {atom: xyz}
    order = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(('ATOM  ', 'HETATM')):
                if line.startswith('ENDMDL'):
                    break          # 只取第一个 model (NMR 结构有多个)
                continue
            atom = line[12:16].strip()
            if atom not in ('N', 'CA', 'C', 'O'):
                continue
            alt = line[16]
            if alt not in (' ', 'A'):      # 只取主构象
                continue
            resname = line[17:20].strip()
            if resname not in THREE2ONE:
                continue
            ch = line[21]
            if chain is not None and ch != chain:
                continue
            key = (ch, line[22:26].strip(), line[26])
            if key not in residues:
                residues[key] = {'aa': THREE2ONE[resname]}
                order.append(key)
            try:
                residues[key][atom] = (
                    float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue

    if chain is None and order:
        first = order[0][0]                        # 只保留第一条链
        order = [k for k in order if k[0] == first]

    # 必须有 N/CA/C 才能建骨架坐标系; O 缺失补零 (与上游行为一致)
    order = [k for k in order if all(a in residues[k] for a in ('N', 'CA', 'C'))]
    if not order:
        raise ValueError(f"no backbone residues in {pdb_path} chain={chain}")

    out = {a: np.array([residues[k].get(a, (0., 0., 0.)) for k in order],
                       dtype=np.float32) for a in ('N', 'CA', 'C', 'O')}
    out['sequence'] = ''.join(residues[k]['aa'] for k in order)
    out['L'] = len(order)
    return out


# ============================================================
# 2. 路径解析
# ============================================================

def resolve_structure(protein_id):
    """protein_id -> (pdb_path, chain, cache_key); 找不到返回 (None, None, key)"""
    pid = str(protein_id)
    if UNIPROT_RE.fullmatch(pid):
        p = os.path.join(AF_DIR, f"AF-{pid}-F1.pdb")
        return (p if os.path.exists(p) else None), None, pid
    # PDB255: 形如 2MA1A = 4 位 PDB ID + 1 位链号
    p = os.path.join(PDB255_DIR, f"{pid[:4].upper()}.pdb")
    return (p if os.path.exists(p) else None), (pid[4] if len(pid) > 4 else None), pid


# ============================================================
# 3. 残基对齐
# ============================================================

def build_res2struct(struct_seq, full_seq, max_len):
    """
    返回 (res2struct, n_matched)
      res2struct: (Lr,) int64, res2struct[j] = 全序列残基 j 对应的结构下标, 无则 -1
      Lr = min(len(full_seq), max_len)

    快路径: 结构序列是全序列的前缀 (AlphaFold 绝大多数情况)。
    慢路径: difflib 找最长匹配块, 处理 RCSB 的缺失残基。
    """
    Lr = min(len(full_seq), max_len)
    res2struct = np.full(Lr, -1, dtype=np.int64)
    Ls = len(struct_seq)

    if Ls and full_seq[:Ls] == struct_seq:              # 完全一致
        n = min(Ls, Lr)
        res2struct[:n] = np.arange(n)
        return res2struct, n

    sm = difflib.SequenceMatcher(None, full_seq[:Lr], struct_seq, autojunk=False)
    n = 0
    for blk in sm.get_matching_blocks():
        if blk.size == 0:
            continue
        res2struct[blk.a:blk.a + blk.size] = np.arange(blk.b, blk.b + blk.size)
        n += blk.size
    return res2struct, n


# ============================================================
# 4. 特征缓存
# ============================================================

def _extract(pdb_path, chain, max_len):
    """解析 + 提特征。延迟 import 上游模块, 避免多进程重复加载。"""
    import sys
    if "/root/DRBP/final/classification" not in sys.path:
        sys.path.insert(0, "/root/DRBP/final/classification")
    from data_processing import extract_structure_features

    bb = parse_backbone(pdb_path, chain)
    feat = extract_structure_features(bb, max_len=max_len, include_pair=False)
    out = {k: torch.as_tensor(np.asarray(feat[k])) for k in STRUCT_KEYS if k in feat}
    out['sequence'] = feat.get('sequence', bb['sequence'])
    return out


def cache_path(key):
    return os.path.join(CACHE_DIR, f"{key}.pt")


def load_or_build(protein_id, max_len, build=True):
    """
    返回结构特征 dict (含 'sequence'), 或 None 表示无结构 (缺文件 / 解析失败)。
    """
    path, chain, key = resolve_structure(protein_id)
    cp = cache_path(key)
    if os.path.exists(cp):
        try:
            return torch.load(cp, map_location='cpu', weights_only=False)
        except Exception:
            pass          # 缓存损坏则重建
    if path is None or not build:
        return None
    try:
        feat = _extract(path, chain, max_len)
    except Exception:
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = cp + f".{os.getpid()}.part"
    torch.save(feat, tmp)
    os.replace(tmp, cp)          # 原子写, 多进程安全
    return feat


def make_empty_struct():
    """无结构时的占位。Ls=1 且 mask 全 0, GNN 输出会被 mask 乘成精确的 0。"""
    return {
        'aa_indices': torch.zeros(1, dtype=torch.long),
        'dihedral_sincos': torch.zeros(1, 12),
        'backbone_frames_R': torch.eye(3).unsqueeze(0),
        'backbone_frames_t': torch.zeros(1, 3),
        'backbone_frames_quat': torch.tensor([[1., 0., 0., 0.]]),
        'backbone_geom': torch.zeros(1, 7),
        'local_atom_coords': torch.zeros(1, 12),
    }


# ============================================================
# 5. collate
# ============================================================

def collate_struct(items, max_res):
    """
    items: list of (struct_feat_dict_or_None, res2struct ndarray)
    返回结构 batch dict + res2struct (B, max_res)

    不用上游的 structure_collate_fn —— 它无条件把 mask[i,:L]=1.0, 会覆盖掉
    空结构的零 mask。
    """
    B = len(items)
    Ls = max(1, max((f['aa_indices'].shape[0] for f, _ in items if f is not None),
                    default=1))
    out = {
        'aa_indices': torch.zeros(B, Ls, dtype=torch.long),
        'dihedral_sincos': torch.zeros(B, Ls, 12),
        'backbone_frames_R': torch.eye(3).repeat(B, Ls, 1, 1),
        'backbone_frames_t': torch.zeros(B, Ls, 3),
        'backbone_frames_quat': torch.tensor([1., 0., 0., 0.]).repeat(B, Ls, 1),
        'backbone_geom': torch.zeros(B, Ls, 7),
        'local_atom_coords': torch.zeros(B, Ls, 12),
        'mask': torch.zeros(B, Ls),
    }
    r2s = torch.full((B, max_res), -1, dtype=torch.long)

    for i, (f, m) in enumerate(items):
        if f is not None:
            L = f['aa_indices'].shape[0]
            for k in STRUCT_KEYS:
                if k in f:
                    out[k][i, :L] = f[k].float() if out[k].dtype == torch.float32 else f[k]
            out['mask'][i, :L] = 1.0
        if m is not None:
            n = min(len(m), max_res)
            r2s[i, :n] = torch.from_numpy(np.ascontiguousarray(m[:n]))
    out['aa_indices'] = out['aa_indices'].long()
    return out, r2s
