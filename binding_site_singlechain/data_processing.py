#!/usr/bin/env python3
"""
DRBP-Net 数据处理模块 (La-Proteina 风格增强版)
================================================

📋 模块总览
-----------
本模块负责从蛋白质结构文件 (PDB) 中提取丰富的几何与拓扑特征，供
StructureEncoder 模型使用。同时提供 ESM-2 序列数据集和结构数据集类。

数据流向
--------
  PDB 文件
    │
    ▼
  parse_pdb_backbone()           ← 解析骨架原子坐标 (N, CA, C, O)
    │
    ▼
  extract_structure_features()   ← 一站式特征提取
    │
    ├── Single 特征 (每残基) ─────────────────────────────
    │   ├── aa_indices          (L,)        AA 类型索引 [0-20]
    │   ├── dihedral_sincos     (L, 12)     二面角 sin/cos 编码
    │   ├── backbone_frames_R   (L, 3, 3)   骨架旋转矩阵
    │   ├── backbone_frames_quat(L, 4)      骨架四元数
    │   ├── backbone_geom       (L, 7)      键长(4) + 键角余弦(3)
    │   └── local_atom_coords   (L, 12)     局部坐标系原子坐标
    │
    └── Pair 特征 (每对残基) ─────────────────────────────
        ├── rel_seq_pos         (L, L)      序列间隔 |i-j|
        ├── pair_dists          (L, L, 16)  全原子对距离 (4×4)
        ├── rel_orient_matrix   (L, L, 12)  相对旋转(9) + 平移(3)
        └── rel_orient_quat     (L, L, 7)   相对四元数(4) + 平移(3)

    │
    ▼
  ProteinStructureDataset        ← PyTorch Dataset, 批量加载 PDB
    │
    ▼
  structure_collate_fn()         ← Batch padding (变长 → 等长)
    │
    ▼
  StructureEncoder (model.py)    ← 模型编码器

特征编码对照表 (data_processing → model.py)
───────────────────────────────────────────
| 数据处理输出                     | 模型输入参数                     | 维度           |
|---------------------------------|---------------------------------|----------------|
| features['aa_indices']          | aa_indices                      | (B, L)         |
| features['dihedral_sincos']     | dihedral_sincos                 | (B, L, 12)     |
| features['backbone_frames_R']   | backbone_frames_R               | (B, L, 3, 3)   |
| features['backbone_frames_quat']| backbone_frames_quat            | (B, L, 4)      |
| features['backbone_geom']       | backbone_geom                   | (B, L, 7)      |
| features['local_atom_coords']   | local_atom_coords               | (B, L, 12)     |
| features['rel_seq_pos']         | rel_seq_pos                     | (B, L, L)      |
| features['pair_dists']          | pair_dists                      | (B, L, L, 16)  |
| features['rel_orient_matrix']   | rel_orient_matrix               | (B, L, L, 12)  |
| features['rel_orient_quat']     | rel_orient_quat                 | (B, L, L, 7)   |
| features['mask']                | mask                            | (B, L)         |

各特征的计算函数
────────────────
| 函数                              | 产出特征                          |
|-----------------------------------|----------------------------------|
| extract_backbone_dihedrals()      | dihedral_angles (φ, ψ, ω)       |
| encode_dihedral_sincos()          | dihedral_sincos (sin/cos 编码)   |
| build_backbone_frames()           | backbone_frames_R, _t, _quat    |
| compute_backbone_bond_geometry()  | backbone_geom (键长+键角)        |
| convert_to_local_coordinates()    | local_atom_coords               |
| compute_all_backbone_pair_distances()| pair_dists (16 通道)           |
| compute_relative_orientation_features()| rel_orient_matrix            |
| compute_relative_quaternion_features()| rel_orient_quat              |
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, List, Optional

# ============================================================
# 氨基酸映射
# ============================================================

AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

# 20 标准氨基酸 + unknown (X)
AA_VOCAB = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
AA_TO_IDX['X'] = 20  # unknown
NAA_TYPES = 21


# ============================================================
# PDB 解析 (增强版: 额外提取 O 原子)
# ============================================================

def parse_pdb_backbone(pdb_path: str) -> Dict:
    """
    解析 PDB 文件，提取骨架原子 (N, CA, C, O) 坐标。

    处理策略:
      - 只取第一个 MODEL (NMR 结构)
      - 只取标准 20 种氨基酸 (跳过 HETATM/水/配体等)
      - 跳过多重构象 (altloc != ' ' and != 'A')
      - 保留所有链，按出现顺序拼接
      - 缺失骨架原子的残基被跳过 (至少需要 N, CA, C)

    Returns:
        coords:    dict {'N': (L,3), 'CA': (L,3), 'C': (L,3), 'O': (L,3)}
                   O 可能为 None 的填充零向量
        sequence:  1-letter 氨基酸序列
        L:         残基数
    """
    with open(pdb_path, 'r') as f:
        lines = f.readlines()

    # 第一遍: 按残基分组收集原子
    residue_data = {}
    residue_order = []

    for line in lines:
        if line.startswith('ENDMDL'):
            break  # 只取第一个 MODEL

        if not line.startswith('ATOM '):
            continue

        atom_name = line[12:16].strip()
        if atom_name not in ('N', 'CA', 'C', 'O', 'CB'):
            continue

        altloc = line[16]
        if altloc not in (' ', 'A'):
            continue  # 跳过多重构象

        res_name = line[17:20].strip()
        if res_name not in AA_3TO1:
            continue

        chain_id = line[21]
        resseq = line[22:26].strip()
        icode = line[26] if len(line) > 26 else ' '
        res_key = (chain_id, resseq, icode)

        if res_key not in residue_data:
            residue_data[res_key] = {
                'N': None, 'CA': None, 'C': None, 'O': None, 'CB': None,
                'aa': res_name
            }
            residue_order.append(res_key)

        # 只取第一个出现的该原子
        if residue_data[res_key][atom_name] is None:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            residue_data[res_key][atom_name] = np.array([x, y, z], dtype=np.float32)

    # 第二遍: 收集完整的残基
    N_list, CA_list, C_list, O_list = [], [], [], []
    seq_chars = []
    residue_ids = []  # 与 seq_chars 同序的 (chain_id, resseq) 列表, 用于对齐结合位点标注

    for res_key in residue_order:
        d = residue_data[res_key]
        if d['N'] is not None and d['CA'] is not None and d['C'] is not None:
            N_list.append(d['N'])
            CA_list.append(d['CA'])
            C_list.append(d['C'])
            # O 允许缺失（某些低分辨率结构中可能缺失），填零
            if d['O'] is not None:
                O_list.append(d['O'])
            else:
                O_list.append(np.zeros(3, dtype=np.float32))
            seq_chars.append(AA_3TO1[d['aa']])
            residue_ids.append((res_key[0], res_key[1]))

    N_coords = np.stack(N_list) if N_list else np.zeros((0, 3), dtype=np.float32)
    CA_coords = np.stack(CA_list) if CA_list else np.zeros((0, 3), dtype=np.float32)
    C_coords = np.stack(C_list) if C_list else np.zeros((0, 3), dtype=np.float32)
    O_coords = np.stack(O_list) if O_list else np.zeros((0, 3), dtype=np.float32)
    sequence = ''.join(seq_chars)

    return {
        'N': N_coords,
        'CA': CA_coords,
        'C': C_coords,
        'O': O_coords,
        'sequence': sequence,
        'L': len(sequence),
        'residue_ids': residue_ids,
    }


# ============================================================
# 二面角计算
# ============================================================

def compute_dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """
    计算四个连续原子 a-b-c-d 的二面角 (扭转角).

    返回值: [-pi, pi] 弧度
    """
    b1 = a - b
    b2 = c - b
    b3 = d - c

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    if n1_norm < 1e-8 or n2_norm < 1e-8:
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    b2_norm = np.linalg.norm(b2)
    m1 = np.cross(n1, n2)

    x = np.dot(n1, n2)
    y = np.dot(m1, b2) / b2_norm

    return np.arctan2(y, x)


def extract_backbone_dihedrals(N_coords: np.ndarray, CA_coords: np.ndarray,
                                C_coords: np.ndarray) -> np.ndarray:
    """
    从骨架坐标提取每残基的二面角 φ, ψ, ω.

    标准定义:
      φ_i  = dihedral(C_{i-1}, N_i, CA_i, C_i)
      ψ_i  = dihedral(N_i, CA_i, C_i, N_{i+1})
      ω_i  = dihedral(CA_{i-1}, C_{i-1}, N_i, CA_i)

    首残基的 φ, ω 和末残基的 ψ, ω 设为 0。

    Args:
        N_coords:   (L, 3)
        CA_coords:  (L, 3)
        C_coords:   (L, 3)

    Returns:
        dihedrals: (L, 3) — [φ, ψ, ω] 每行
    """
    L = N_coords.shape[0]
    dihedrals = np.zeros((L, 3), dtype=np.float32)

    for i in range(L):
        # φ_i (phi): C_{i-1} - N_i - CA_i - C_i
        if i > 0:
            dihedrals[i, 0] = compute_dihedral(
                C_coords[i-1], N_coords[i], CA_coords[i], C_coords[i]
            )

        # ψ_i (psi): N_i - CA_i - C_i - N_{i+1}
        if i < L - 1:
            dihedrals[i, 1] = compute_dihedral(
                N_coords[i], CA_coords[i], C_coords[i], N_coords[i+1]
            )

        # ω_i (omega): CA_{i-1} - C_{i-1} - N_i - CA_i
        if i > 0:
            dihedrals[i, 2] = compute_dihedral(
                CA_coords[i-1], C_coords[i-1], N_coords[i], CA_coords[i]
            )

    return dihedrals


# ============================================================
# 骨架刚体框架 (Backbone Rigid-Body Frames)
# ============================================================

def build_backbone_frames(N_coords: np.ndarray, CA_coords: np.ndarray,
                           C_coords: np.ndarray) -> Dict:
    """
    为每个残基构建骨架局部坐标系 (SE(3) 刚体框架).

    采用 AlphaFold/La-Proteina 标准:
      原点: CA
      x 轴: (C - CA) / |C - CA|           # 沿 CA→C 方向
      y 轴: (N - CA) 投影到 x 的垂直平面，归一化
      z 轴: x × y

    这给每个残基一个正交的右手坐标系 (rotation matrix + translation).

    Args:
        N_coords:  (L, 3)
        CA_coords: (L, 3)
        C_coords:  (L, 3)

    Returns:
        frames: dict with:
          'R':        (L, 3, 3)  旋转矩阵
          't':        (L, 3)     平移向量 (= CA 坐标)
          'quat':     (L, 4)     四元数表示 (w, x, y, z)
          'translation': (L, 3)  同 t, 用于命名一致性
    """
    L = CA_coords.shape[0]

    # x 轴: CA → C
    x_axis = C_coords - CA_coords  # (L, 3)
    x_norm = np.linalg.norm(x_axis, axis=-1, keepdims=True) + 1e-8
    x_axis = x_axis / x_norm

    # y 轴: (N - CA) 的 x-垂直分量
    v_n = N_coords - CA_coords  # (L, 3)
    # 投影到 x 的垂直平面: y_raw = v_n - (v_n · x) * x
    dot_vn_x = np.sum(v_n * x_axis, axis=-1, keepdims=True)
    y_axis = v_n - dot_vn_x * x_axis
    y_norm = np.linalg.norm(y_axis, axis=-1, keepdims=True) + 1e-8
    y_axis = y_axis / y_norm

    # z 轴: x × y
    z_axis = np.cross(x_axis, y_axis)
    # 确保单位长度
    z_norm = np.linalg.norm(z_axis, axis=-1, keepdims=True) + 1e-8
    z_axis = z_axis / z_norm

    # 重新正交化 y (确保完全正交): y = z × x
    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis, axis=-1, keepdims=True) + 1e-8
    y_axis = y_axis / y_norm

    # 组装旋转矩阵 (L, 3, 3): 每行是坐标轴在世界坐标下的分量
    R = np.stack([x_axis, y_axis, z_axis], axis=-1)  # (L, 3, 3)

    # 确保 det(R) = +1 (右手系)
    det = np.linalg.det(R)
    R[:, :, 2] *= np.sign(det)[:, None]

    # 四元数
    quat = rotation_matrix_to_quaternion(R)

    return {
        'R': R.astype(np.float32),
        't': CA_coords.astype(np.float32),
        'quat': quat.astype(np.float32),
    }


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    旋转矩阵 → 四元数 (w, x, y, z).

    Args:
        R: (L, 3, 3) 或 (3, 3)

    Returns:
        quat: (L, 4) 或 (4,)
    """
    if R.ndim == 2:
        R = R[None, ...]
        squeeze = True
    else:
        squeeze = False

    L = R.shape[0]
    quat = np.zeros((L, 4), dtype=np.float32)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    for i in range(L):
        r = R[i]
        t = trace[i]
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            quat[i, 0] = 0.25 * s  # w
            quat[i, 1] = (r[2, 1] - r[1, 2]) / s  # x
            quat[i, 2] = (r[0, 2] - r[2, 0]) / s  # y
            quat[i, 3] = (r[1, 0] - r[0, 1]) / s  # z
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            quat[i, 0] = (r[2, 1] - r[1, 2]) / s
            quat[i, 1] = 0.25 * s
            quat[i, 2] = (r[0, 1] + r[1, 0]) / s
            quat[i, 3] = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            quat[i, 0] = (r[0, 2] - r[2, 0]) / s
            quat[i, 1] = (r[0, 1] + r[1, 0]) / s
            quat[i, 2] = 0.25 * s
            quat[i, 3] = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            quat[i, 0] = (r[1, 0] - r[0, 1]) / s
            quat[i, 1] = (r[0, 2] + r[2, 0]) / s
            quat[i, 2] = (r[1, 2] + r[2, 1]) / s
            quat[i, 3] = 0.25 * s

    if squeeze:
        quat = quat[0]
    return quat


# ============================================================
# 骨架键长/键角几何
# ============================================================

def compute_backbone_bond_geometry(N_coords: np.ndarray, CA_coords: np.ndarray,
                                    C_coords: np.ndarray) -> np.ndarray:
    """
    计算每残基的局部骨架几何特征 (键长 + 键角).

    Features (共 9 维):
      键长:
        d_CA_N   = |N_i - CA_i|        # N-CA 键长
        d_CA_C   = |C_i - CA_i|        # CA-C 键长
        d_C_N1   = |N_{i+1} - C_i|     # 肽键 (C-N)
        d_N_CA1  = |CA_{i+1} - N_{i+1}|  # 下一个 N-CA

      键角 (余弦值):  # 用余弦而非角度值，避免周期性
        cos_N_CA_C   = angle(N_i, CA_i, C_i)
        cos_CA_C_N1  = angle(CA_i, C_i, N_{i+1})
        cos_C_N1_CA1 = angle(C_i, N_{i+1}, CA_{i+1})

      肽平面扭转补充:
        psi_prev = ψ_{i-1} 的补充信息
        phi_next = φ_{i+1} 的补充信息

    Args:
        N_coords:  (L, 3)
        CA_coords: (L, 3)
        C_coords:  (L, 3)

    Returns:
        geom: (L, 7) 键长(4) + 键角余弦(3)
    """
    L = N_coords.shape[0]
    geom = np.zeros((L, 7), dtype=np.float32)

    for i in range(L):
        # 键长
        d_ca_n = np.linalg.norm(N_coords[i] - CA_coords[i])
        d_ca_c = np.linalg.norm(C_coords[i] - CA_coords[i])
        geom[i, 0] = d_ca_n
        geom[i, 1] = d_ca_c

        if i < L - 1:
            d_c_n1 = np.linalg.norm(N_coords[i+1] - C_coords[i])
            d_n_ca1 = np.linalg.norm(CA_coords[i+1] - N_coords[i+1])
            geom[i, 2] = d_c_n1
            geom[i, 3] = d_n_ca1
        else:
            geom[i, 2] = geom[i-1, 2] if i > 0 else 1.33  # 典型肽键长度
            geom[i, 3] = geom[i-1, 3] if i > 0 else 1.46  # 典型 N-CA 长度

        # 键角 (余弦)
        # N-CA-C
        v1 = N_coords[i] - CA_coords[i]
        v2 = C_coords[i] - CA_coords[i]
        geom[i, 4] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

        if i < L - 1:
            # CA-C-N_{i+1}
            v1 = CA_coords[i] - C_coords[i]
            v2 = N_coords[i+1] - C_coords[i]
            geom[i, 5] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

            # C-N_{i+1}-CA_{i+1}
            v1 = C_coords[i] - N_coords[i+1]
            v2 = CA_coords[i+1] - N_coords[i+1]
            geom[i, 6] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        else:
            geom[i, 5] = geom[i-1, 5] if i > 0 else np.cos(np.deg2rad(121))
            geom[i, 6] = geom[i-1, 6] if i > 0 else np.cos(np.deg2rad(116))

    return geom


# ============================================================
# 全骨架原子对距离矩阵
# ============================================================

def compute_all_backbone_pair_distances(N_coords: np.ndarray, CA_coords: np.ndarray,
                                         C_coords: np.ndarray,
                                         O_coords: Optional[np.ndarray] = None) -> np.ndarray:
    """
    计算所有骨架原子对之间的欧氏距离矩阵.

    对于每对残基 (i, j)，计算:
      d(N_i, N_j),  d(N_i, CA_j),  d(N_i, C_j),  d(N_i, O_j)
      d(CA_i, N_j), d(CA_i, CA_j), d(CA_i, C_j), d(CA_i, O_j)
      d(C_i, N_j),  d(C_i, CA_j),  d(C_i, C_j),  d(C_i, O_j)
      d(O_i, N_j),  d(O_i, CA_j),  d(O_i, C_j),  d(O_i, O_j)

    这些比仅用 CA-CA 距离提供了更丰富的结构信息。

    Args:
        N_coords:  (L, 3)
        CA_coords: (L, 3)
        C_coords:  (L, 3)
        O_coords:  (L, 3) optional, 若 None 则跳过 O 相关

    Returns:
        pair_dists: (L, L, n_channels)
          - O 存在时: 16 通道 (4×4)
          - O 不存在时: 9 通道 (3×3)
    """
    L = N_coords.shape[0]

    backbone_atoms = [N_coords, CA_coords, C_coords]
    atom_names = ['N', 'CA', 'C']
    if O_coords is not None:
        backbone_atoms.append(O_coords)
        atom_names.append('O')

    n_atoms = len(backbone_atoms)
    pair_dists = np.zeros((L, L, n_atoms * n_atoms), dtype=np.float32)

    for a in range(n_atoms):
        for b in range(n_atoms):
            channel_idx = a * n_atoms + b
            diff = backbone_atoms[a][:, None, :] - backbone_atoms[b][None, :, :]  # (L, L, 3)
            dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-8)
            pair_dists[:, :, channel_idx] = dist

    return pair_dists


# ============================================================
# 残基对间相对朝向特征
# ============================================================

def compute_relative_orientation_features(frames: Dict, L: int) -> np.ndarray:
    """
    计算每对残基之间的相对 SE(3) 变换特征.

    对每个 (i, j):
      - 相对旋转: R_{ij} = R_i^T @ R_j  → 平坦化为 9 维
      - 相对平移: t_{ij} = R_i^T @ (t_j - t_i)  → 3 维 (在 i 的局部坐标系中)

    La-Proteina 中，这些相对朝向特征被编码到 pair representation 中，
    提供残基间空间相对位置和朝向的完整信息。

    Args:
        frames: build_backbone_frames() 的输出
        L:      残基数

    Returns:
        rel_orient: (L, L, 12)  相对旋转(9) + 相对平移(3)
    """
    R = frames['R']   # (L, 3, 3)
    t = frames['t']   # (L, 3)

    rel_orient = np.zeros((L, L, 12), dtype=np.float32)

    for i in range(L):
        for j in range(L):
            if i == j:
                # 自身: 旋转=单位阵, 平移=0
                rel_orient[i, j, 0] = 1.0
                rel_orient[i, j, 4] = 1.0
                rel_orient[i, j, 8] = 1.0
                continue

            # 相对旋转: R_i^T @ R_j (3x3, 平坦化)
            R_rel = R[i].T @ R[j]  # (3, 3)
            rel_orient[i, j, :9] = R_rel.flatten()

            # 相对平移: R_i^T @ (t_j - t_i)  (在 i 的局部坐标中)
            t_rel = R[i].T @ (t[j] - t[i])
            rel_orient[i, j, 9:12] = t_rel

    return rel_orient


def compute_relative_quaternion_features(frames: Dict, L: int) -> np.ndarray:
    """
    用四元数表示相对朝向 (更紧凑).

    Args:
        frames: build_backbone_frames() 的输出
        L:      残基数

    Returns:
        rel_quat: (L, L, 7)  相对四元数(4) + 相对平移(3)
    """
    quat = frames['quat']  # (L, 4)
    t = frames['t']        # (L, 3)

    rel_quat = np.zeros((L, L, 7), dtype=np.float32)

    for i in range(L):
        for j in range(L):
            # 相对四元数: q_j * conj(q_i)
            q_i = quat[i]
            q_j = quat[j]
            # conj(q_i) = (w, -x, -y, -z)
            q_i_conj = np.array([q_i[0], -q_i[1], -q_i[2], -q_i[3]])
            q_rel = quaternion_multiply(q_j, q_i_conj)
            rel_quat[i, j, :4] = q_rel

            # 相对平移
            t_rel = t[j] - t[i]
            rel_quat[i, j, 4:7] = t_rel

    return rel_quat


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法 (w1, x1, y1, z1) * (w2, x2, y2, z2)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
    ], dtype=np.float32)


# ============================================================
# 特征编码
# ============================================================

def encode_dihedral_sincos(angles: np.ndarray, n_freqs: int = 2) -> np.ndarray:
    """
    对二面角做 sin/cos 多频率编码.

    Args:
        angles:  (L, 3) 弧度值
        n_freqs: 频率数 (1x, 2x, ...)

    Returns:
        encoded: (L, 3 * n_freqs * 2)
    """
    L = angles.shape[0]
    encoded = []
    for f in range(1, n_freqs + 1):
        encoded.append(np.sin(f * angles))
        encoded.append(np.cos(f * angles))
    return np.concatenate(encoded, axis=1).astype(np.float32)


def sincos_encode_angles(angles: torch.Tensor, n_freqs: int = 2) -> torch.Tensor:
    """
    对角度的 sin/cos 多频率编码 (PyTorch 版).

    在模型内部使用，与 numpy 版的 encode_dihedral_sincos 功能相同。

    Args:
        angles:  (*, n_angles)  弧度
        n_freqs: 频率数 (1x, 2x, ...)

    Returns:
        encoded: (*, n_angles * n_freqs * 2)
    """
    encoded = []
    for f in range(1, n_freqs + 1):
        encoded.append(torch.sin(f * angles))
        encoded.append(torch.cos(f * angles))
    return torch.cat(encoded, dim=-1)


# ============================================================
# 局部坐标变换
# ============================================================

def convert_to_local_coordinates(N_coords: np.ndarray, CA_coords: np.ndarray,
                                   C_coords: np.ndarray, O_coords: np.ndarray,
                                   frames: Dict) -> np.ndarray:
    """
    将所有骨架原子转换到各残基的局部坐标系中.

    这提供每个残基骨架原子的 "规范姿态" 表示，对局部结构变异敏感。

    Args:
        N_coords:  (L, 3)
        CA_coords: (L, 3)
        C_coords:  (L, 3)
        O_coords:  (L, 3)
        frames:    build_backbone_frames() 的输出

    Returns:
        local_coords: (L, 12)  [N_loc, CA_loc, C_loc, O_loc] 每原子 3 维
    """
    L = N_coords.shape[0]
    R = frames['R']   # (L, 3, 3)
    t = frames['t']   # (L, 3)

    atoms = [N_coords, CA_coords, C_coords, O_coords]
    local_concat = []

    for atom_coords in atoms:
        if atom_coords is None: continue
        # 转换到局部坐标: R_i^T @ (atom - t_i)
        local = np.einsum('lij,lj->li', R.transpose(0, 2, 1), atom_coords - t)
        local_concat.append(local)

    return np.concatenate(local_concat, axis=-1).astype(np.float32)  # (L, 12)


# ============================================================
# 结构特征提取 (La-Proteina 风格)
# ============================================================

def compute_ca_neighbor_count(CA_coords: np.ndarray, cutoff: float = 12.0) -> np.ndarray:
    """每个残基的 Cα 邻居数(配位数), 代理表面暴露度: 少=暴露, 多=埋藏。

    Returns:
        (L,) float32, 每个残基 cutoff Å 内的 Cα 邻居个数(不含自身)
    """
    diff = CA_coords[:, None, :] - CA_coords[None, :, :]  # (L, L, 3)
    dists = np.sqrt((diff ** 2).sum(-1))                  # (L, L)
    np.fill_diagonal(dists, np.inf)                        # 排除自身
    count = (dists < cutoff).sum(axis=1).astype(np.float32)  # (L,)
    return count


def _fibonacci_sphere(n: int) -> np.ndarray:
    """单位球面均匀撒 n 个点 (黄金螺旋法)。"""
    pts = np.zeros((n, 3), dtype=np.float32)
    phi = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2
        r = np.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        pts[i] = [np.cos(theta) * r, y, np.sin(theta) * r]
    return pts


def compute_ca_sasa(CA_coords: np.ndarray, radius: float = 3.5, probe: float = 1.4,
                    n_points: int = 60) -> np.ndarray:
    """Cα 级 Shrake-Rupley SASA: 每个 Cα 放一个球, 用 1.4Å 水探针算暴露面积。

    替代旧的 ca_exposure(Cα 邻居计数代理)。这是标准 Shrake-Rupley 算法(与 DSSP 同源),
    只是用 Cα 球代替全原子, 返回 [0,1] 暴露度 (0=埋藏, 1=全暴露)。
    """
    L = CA_coords.shape[0]
    pts = _fibonacci_sphere(n_points)
    R = radius + probe
    sasa = np.zeros(L, dtype=np.float32)
    for i in range(L):
        surf = CA_coords[i] + pts * R
        d = np.linalg.norm(surf[:, None, :] - CA_coords[None, :, :], axis=-1)  # (P, L)
        covered = d < R
        covered[:, i] = False
        sasa[i] = (1.0 - covered.any(axis=1).mean()) * 4 * np.pi * R ** 2
    return sasa / (4 * np.pi * R ** 2)   # 归一化到 [0,1]


def compute_ca_concavity(CA_coords: np.ndarray, cutoff: float = 12.0) -> np.ndarray:
    """凹凸度代理: 残基径向距离 vs 邻居平均径向距离之差。

    正 = 凹(比邻居更深, 在凹槽/口袋里); 负 = 凸(比邻居更浅, 凸起)。
    """
    center = CA_coords.mean(axis=0)                        # 全局质心
    radial = np.linalg.norm(CA_coords - center, axis=1)    # (L,)
    diff = CA_coords[:, None, :] - CA_coords[None, :, :]
    dists = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dists, np.inf)
    L = CA_coords.shape[0]
    concavity = np.zeros(L, dtype=np.float32)
    for i in range(L):
        nb = dists[i] < cutoff
        if nb.sum() > 0:
            concavity[i] = radial[nb].mean() - radial[i]
    return concavity


AA_CHARGE = {'K': 1.0, 'R': 1.0, 'H': 0.5, 'D': -1.0, 'E': -1.0}


def compute_ca_electrostatics(CA_coords: np.ndarray, sequence: str, cutoff: float = 15.0) -> np.ndarray:
    """静电势代理: 每个残基附近的正/负电荷加权和 (库仑势近似, 无 APBS)。

    正 = 附近偏正电(利于结合负电的 DNA 骨架); 负 = 附近偏负电。
    """
    L = len(sequence)
    charges = np.array([AA_CHARGE.get(aa, 0.0) for aa in sequence], dtype=np.float32)  # (L,)
    diff = CA_coords[:, None, :] - CA_coords[None, :, :]
    dists = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dists, np.inf)
    inv_d = np.where(dists < cutoff, 1.0 / (dists + 1e-6), 0.0)
    potential = (charges[None, :] * inv_d).sum(axis=1).astype(np.float32)  # (L,)
    return potential


def extract_structure_features(pdb_result: Dict, max_len: int = 512,
                                 include_pair: bool = True) -> Dict:
    """
    从 PDB 解析结果中提取完整结构特征。

    特征组成 (La-Proteina 风格):

    *** Single Features (每残基, 用于 single representation) ***
      - aa_indices:          (L,)   AA 类型索引 [0-20]
      - dihedral_angles:     (L, 3) 二面角 (φ, ψ, ω) 弧度
      - dihedral_sincos:     (L, d_dih_enc) sin/cos 编码的二面角
      - backbone_frames_R:   (L, 3, 3) 骨架局部旋转矩阵
      - backbone_frames_t:   (L, 3)   骨架局部平移 (= CA 坐标)
      - backbone_frames_quat:(L, 4)   骨架四元数
      - backbone_geom:       (L, 7)   骨架键长和键角
      - local_atom_coords:   (L, 12)  局部坐标系中的原子坐标

    *** Pair Features (每对残基, 用于 pair representation) ***
      - rel_seq_sep:         (L, L)   序列间隔 |i-j|
      - pair_dists:          (L, L, n_ch) 全原子对距离 (最多 16 通道)
      - rel_orient_matrix:   (L, L, 12) 相对 SE(3) 变换 (旋转平坦化 + 平移)
      - rel_orient_quat:     (L, L, 7)  相对四元数 + 平移

    注意: pair 特征在 La-Proteina 中通过 Pair-Biased Attention 和
    Outer Product Mean 来更新 single 表征。最终的 single 特征包含
    了从 pair 传递过来的结构信息。

    Args:
        pdb_result: parse_pdb_backbone() 的返回值
        max_len:    最大序列长度（截断）
        include_pair: 是否计算 pair 特征 (训练时需要, 预测时可能不需要)

    Returns:
        feature dict
    """
    L = pdb_result['L']
    N_coords = pdb_result['N']
    CA_coords = pdb_result['CA']
    C_coords = pdb_result['C']
    O_coords = pdb_result['O']
    sequence = pdb_result['sequence']

    # 截断
    if L > max_len:
        L = max_len
        N_coords = N_coords[:L]
        CA_coords = CA_coords[:L]
        C_coords = C_coords[:L]
        O_coords = O_coords[:L]
        sequence = sequence[:L]

    # ---- AA 索引 ----
    aa_indices = np.array([AA_TO_IDX.get(aa, 20) for aa in sequence], dtype=np.int64)

    # ---- 二面角 ----
    dihedral_angles = extract_backbone_dihedrals(N_coords, CA_coords, C_coords)
    dihedral_sincos = encode_dihedral_sincos(dihedral_angles, n_freqs=2)  # (L, 12)

    # ---- 骨架刚体框架 ----
    frames = build_backbone_frames(N_coords, CA_coords, C_coords)

    # ---- 骨架键长键角 ----
    backbone_geom = compute_backbone_bond_geometry(N_coords, CA_coords, C_coords)

    # ---- 局部坐标 ----
    local_coords = convert_to_local_coordinates(N_coords, CA_coords, C_coords, O_coords, frames)

    # ---- 表面特征 (SASA + 凹凸度 + 静电) ----
    ca_exposure = compute_ca_sasa(CA_coords)                    # (L,) Shrake-Rupley Cα级 SASA [0,1]
    ca_concavity = compute_ca_concavity(CA_coords)              # (L,) 正=凹, 负=凸
    ca_electrostatics = compute_ca_electrostatics(CA_coords, sequence)  # (L,) 正=正电环境

    result = {
        # Single 特征
        'aa_indices': aa_indices,
        'dihedral_angles': dihedral_angles,
        'dihedral_sincos': dihedral_sincos,
        'backbone_frames_R': frames['R'],
        'backbone_frames_t': frames['t'],
        'backbone_frames_quat': frames['quat'],
        'backbone_geom': backbone_geom,
        'local_atom_coords': local_coords,
        'ca_exposure': ca_exposure,
        'ca_concavity': ca_concavity,
        'ca_electrostatics': ca_electrostatics,
        # 掩码
        'mask': np.ones(L, dtype=np.float32),
        'sequence': sequence,
    }

    if include_pair:
        # ---- 序列间隔 ----
        rel_seq_pos = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])

        # ---- 全原子对距离 ----
        pair_dists = compute_all_backbone_pair_distances(
            N_coords, CA_coords, C_coords, O_coords
        )

        # ---- 相对朝向 (旋转矩阵平坦化) ----
        rel_orient_matrix = compute_relative_orientation_features(frames, L)

        # ---- 相对朝向 (四元数, 更紧凑) ----
        rel_orient_quat = compute_relative_quaternion_features(frames, L)

        result.update({
            'rel_seq_pos': rel_seq_pos.astype(np.int64),
            'pair_dists': pair_dists,         # (L, L, 16)
            'rel_orient_matrix': rel_orient_matrix,  # (L, L, 12)
            'rel_orient_quat': rel_orient_quat,      # (L, L, 7)
        })

    return result


# ============================================================
# 数据集: ESM-2 序列模式
# ============================================================

class ProteinSequenceDataset(Dataset):
    """ESM-2 序列数据集 (同原有逻辑)"""

    def __init__(self, sequences: List[str], dbp_labels: List[int],
                 rbp_labels: List[int], tokenizer, max_len: int = 1024):
        self.sequences = sequences
        self.dbp_labels = torch.tensor(dbp_labels, dtype=torch.float32)
        self.rbp_labels = torch.tensor(rbp_labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tokens = self.tokenizer(
            seq,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
        )
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'dbp_label': self.dbp_labels[idx],
            'rbp_label': self.rbp_labels[idx],
        }


# ============================================================
# 数据集: 结构模式 (增强版)
# ============================================================

class ProteinStructureDataset(Dataset):
    """
    蛋白质结构数据集 (La-Proteina 风格).

    从 PDB 文件和标签 CSV 加载数据，提取丰富的 Single + Pair 特征:
      *** Single ***
        - AA 类型索引
        - 二面角 sin/cos 编码 (φ, ψ, ω)
        - 骨架刚体框架 (旋转矩阵 + 四元数 + 平移)
        - 骨架键长键角
        - 局部坐标系原子坐标

      *** Pair ***
        - 残基对全原子距离矩阵 (16 通道)
        - 相对 SE(3) 变换 (旋转平坦化)
        - 相对四元数 + 相对平移
        - 序列间隔
    """

    def __init__(self, pdb_dir: str, label_df, max_len: int = 512):
        """
        Args:
            pdb_dir:  存放 .pdb 文件的目录
            label_df: pandas DataFrame，必须有列 pdb_id, dbp_label, rbp_label
            max_len:  最大序列长度
        """
        self.pdb_dir = pdb_dir
        self.max_len = max_len

        # 建索引: pdb_id -> label
        self.labels = {}
        for _, row in label_df.iterrows():
            pdb_id = str(row['pdb_id']).strip()
            self.labels[pdb_id] = (
                float(row['dbp_label']),
                float(row['rbp_label']),
            )

        self.pdb_ids = list(self.labels.keys())
        self._cache = {}

    def __len__(self):
        return len(self.pdb_ids)

    def __getitem__(self, idx):
        pdb_id = self.pdb_ids[idx]
        dbp_label, rbp_label = self.labels[pdb_id]

        # 缓存: 每个 PDB 只解析一次
        if pdb_id in self._cache:
            return self._cache[pdb_id]

        # 优先读预计算特征 (存为 .pt 文件, fast!)
        precomp_dir = os.path.join(os.path.dirname(self.pdb_dir), 'structure_features')
        precomp_path = os.path.join(precomp_dir, f'{pdb_id}.pt')
        if os.path.exists(precomp_path):
            features = torch.load(precomp_path, map_location='cpu', weights_only=False)
            L = features['aa_indices'].shape[0]
            result = {
                'aa_indices': features['aa_indices'].long(),
                'dihedral_angles': features['dihedral_angles'],
                'dihedral_sincos': features['dihedral_sincos'],
                'backbone_frames_R': features['backbone_frames_R'],
                'backbone_frames_t': features['backbone_frames_t'],
                'backbone_frames_quat': features.get('backbone_frames_quat', torch.zeros(L, 4)),
                'backbone_geom': features.get('backbone_geom', torch.zeros(L, 7)),
                'local_atom_coords': features.get('local_atom_coords', torch.zeros(L, 12)),
                'mask': features['mask'],
                'dbp_label': torch.tensor(dbp_label, dtype=torch.float32),
                'rbp_label': torch.tensor(rbp_label, dtype=torch.float32),
                'pdb_id': pdb_id, 'sequence': features.get('sequence', ''), 'L': L,
            }
            if 'pair_dists' in features:
                result['pair_dists'] = features['pair_dists']
                result['rel_seq_pos'] = features['rel_seq_pos']
                result['rel_orient_matrix'] = features['rel_orient_matrix']
                result['rel_orient_quat'] = features['rel_orient_quat']
            self._cache[pdb_id] = result
            return result

        # 查找 PDB 文件
        pdb_path = None
        for ext in ['.pdb', '.ent', '.pdb.gz']:
            candidate = os.path.join(self.pdb_dir, f"{pdb_id}{ext}")
            if os.path.exists(candidate):
                pdb_path = candidate
                break

        if pdb_path is None:
            raise FileNotFoundError(
                f"PDB file not found for {pdb_id} in {self.pdb_dir}"
            )

        # 解析 PDB 并提取特征
        pdb_result = parse_pdb_backbone(pdb_path)
        features = extract_structure_features(pdb_result, self.max_len, include_pair=False)

        L = features['aa_indices'].shape[0]

        result = {
            'aa_indices': torch.tensor(features['aa_indices'], dtype=torch.long),
            'dihedral_angles': torch.tensor(features['dihedral_angles'], dtype=torch.float32),
            'dihedral_sincos': torch.tensor(features['dihedral_sincos'], dtype=torch.float32),
            'backbone_frames_R': torch.tensor(features['backbone_frames_R'], dtype=torch.float32),
            'backbone_frames_t': torch.tensor(features['backbone_frames_t'], dtype=torch.float32),
            'backbone_frames_quat': torch.tensor(features['backbone_frames_quat'], dtype=torch.float32),
            'backbone_geom': torch.tensor(features['backbone_geom'], dtype=torch.float32),
            'local_atom_coords': torch.tensor(features['local_atom_coords'], dtype=torch.float32),
            'mask': torch.tensor(features['mask'], dtype=torch.float32),
            'dbp_label': torch.tensor(dbp_label, dtype=torch.float32),
            'rbp_label': torch.tensor(rbp_label, dtype=torch.float32),
            'pdb_id': pdb_id,
            'sequence': features['sequence'],
            'L': L,
        }
        # Pair 特征 (可选: 轻量模式不计算)
        if 'pair_dists' in features:
            result['pair_dists'] = torch.tensor(features['pair_dists'], dtype=torch.float32)
            result['rel_seq_pos'] = torch.tensor(features['rel_seq_pos'], dtype=torch.long)
            result['rel_orient_matrix'] = torch.tensor(features['rel_orient_matrix'], dtype=torch.float32)
            result['rel_orient_quat'] = torch.tensor(features['rel_orient_quat'], dtype=torch.float32)
        self._cache[pdb_id] = result
        return result


# ============================================================
# Collate 函数: 结构数据 batch padding
# ============================================================

def structure_collate_fn(batch: List[Dict]) -> Dict:
    """
    对结构数据 batch 做 padding。

    支持 La-Proteina 风格的丰富特征:
      - Single: (B, L_max, d_*)
      - Pair:   (B, L_max, L_max, d_*)
    """
    batch_max_len = max(item['aa_indices'].shape[0] for item in batch)
    batch_size = len(batch)

    # --- Single 特征 ---
    aa_indices = torch.zeros(batch_size, batch_max_len, dtype=torch.long)
    dihedral_angles = torch.zeros(batch_size, batch_max_len, 3)
    dihedral_sincos = torch.zeros(batch_size, batch_max_len,
                                   batch[0]['dihedral_sincos'].shape[-1])
    backbone_frames_R = torch.zeros(batch_size, batch_max_len, 3, 3)
    backbone_frames_t = torch.zeros(batch_size, batch_max_len, 3)
    backbone_frames_quat = torch.zeros(batch_size, batch_max_len, 4)
    backbone_geom = torch.zeros(batch_size, batch_max_len,
                                 batch[0]['backbone_geom'].shape[-1])
    local_atom_coords = torch.zeros(batch_size, batch_max_len,
                                     batch[0]['local_atom_coords'].shape[-1])
    mask = torch.zeros(batch_size, batch_max_len)

    # --- Pair 特征 (可选: 轻量模式不计算 pair) ---
    has_pair = 'pair_dists' in batch[0]
    if has_pair:
        pair_d_dim = batch[0]['pair_dists'].shape[-1]
        pair_dists = torch.zeros(batch_size, batch_max_len, batch_max_len, pair_d_dim)
        rel_seq_pos = torch.zeros(batch_size, batch_max_len, batch_max_len, dtype=torch.long)
        rel_orient_matrix = torch.zeros(batch_size, batch_max_len, batch_max_len,
                                         batch[0]['rel_orient_matrix'].shape[-1])
        rel_orient_quat = torch.zeros(batch_size, batch_max_len, batch_max_len,
                                       batch[0]['rel_orient_quat'].shape[-1])

    # --- 标签等 ---
    dbp_labels = torch.zeros(batch_size)
    rbp_labels = torch.zeros(batch_size)
    pdb_ids = []
    sequences = []

    for i, item in enumerate(batch):
        L = item['L']

        # Single
        aa_indices[i, :L] = item['aa_indices']
        dihedral_angles[i, :L] = item['dihedral_angles']
        dihedral_sincos[i, :L] = item['dihedral_sincos']
        backbone_frames_R[i, :L] = item['backbone_frames_R']
        backbone_frames_t[i, :L] = item['backbone_frames_t']
        backbone_frames_quat[i, :L] = item['backbone_frames_quat']
        backbone_geom[i, :L] = item['backbone_geom']
        local_atom_coords[i, :L] = item['local_atom_coords']
        mask[i, :L] = 1.0

        # Pair (可选)
        if has_pair:
            pair_dists[i, :L, :L] = item['pair_dists']
            rel_seq_pos[i, :L, :L] = item['rel_seq_pos']
            rel_orient_matrix[i, :L, :L] = item['rel_orient_matrix']
            rel_orient_quat[i, :L, :L] = item['rel_orient_quat']

        # Labels
        dbp_labels[i] = item['dbp_label']
        rbp_labels[i] = item['rbp_label']
        pdb_ids.append(item['pdb_id'])
        sequences.append(item['sequence'])

    result = {
        'aa_indices': aa_indices,
        'dihedral_angles': dihedral_angles,
        'dihedral_sincos': dihedral_sincos,
        'backbone_frames_R': backbone_frames_R,
        'backbone_frames_t': backbone_frames_t,
        'backbone_frames_quat': backbone_frames_quat,
        'backbone_geom': backbone_geom,
        'local_atom_coords': local_atom_coords,
        'mask': mask,
        'dbp_label': dbp_labels,
        'rbp_label': rbp_labels,
        'pdb_id': pdb_ids,
        'sequence': sequences,
    }
    if has_pair:
        result['pair_dists'] = pair_dists
        result['rel_seq_pos'] = rel_seq_pos
        result['rel_orient_matrix'] = rel_orient_matrix
        result['rel_orient_quat'] = rel_orient_quat
    return result


# ============================================================
# 工具函数: 相对位置 / 距离分桶 (保留兼容)
# ============================================================

def build_rel_pos_matrix(L: int, max_rel_pos: int = 32) -> np.ndarray:
    """
    构建相对位置矩阵，值域 [0, 2*max_rel_pos]，0 留给 padding。

    Returns:
        rel_pos: (L, L) 其中 rel_pos[i,j] = clip(j-i, -32, 32) + 32
    """
    pos = np.arange(L)
    rel_pos = pos[:, None] - pos[None, :]  # (L, L)
    rel_pos = np.clip(rel_pos, -max_rel_pos, max_rel_pos) + max_rel_pos
    return rel_pos.astype(np.int64)


def build_dist_bins(dist_matrix: np.ndarray, n_bins: int = 64,
                     max_dist: float = 32.0) -> np.ndarray:
    """
    将距离矩阵分桶为离散索引。

    Returns:
        dist_bins: (L, L) 值域 [0, n_bins-1]
    """
    dist_bins = np.clip(dist_matrix / max_dist * n_bins, 0, n_bins - 1)
    return dist_bins.astype(np.int64)
