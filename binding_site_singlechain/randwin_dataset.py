#!/usr/bin/env python3
"""随机窗口 binding site 数据集。

核心思路: 训练时从全长蛋白随机抽一个 `window` 长度的窗口 (覆盖头部/中部/尾部),
让模型学会在蛋白任意位置预测结合位点; 推理时 (eval_sliding.py) 用重叠滑动窗口
覆盖全长再拼接。

结构特征按**全长**提取后再切片, 保证 SASA(真 DSSP) / 凹凸度 / 静电势这些表面
特征是"全局"的 (基于全长结构算), 而不是窗口局部——这比旧版 `[:max_len]` 截断
更正确。k-NN 图在 StructureGNN 内部按窗口 Cα 坐标重建 (接受边界损耗, 靠
滑动重叠 + 中心加权缓解)。

复用 /root/DRBP/binding_site_v2 的 parse_pdb_backbone / compute_dssp_sasa /
StructureGNN / JointBindingSiteModel / binding_site_collate_fn。
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_processing import (
    AA_TO_IDX, parse_pdb_backbone,
    extract_backbone_dihedrals, encode_dihedral_sincos,
    build_backbone_frames, compute_backbone_bond_geometry,
    convert_to_local_coordinates, compute_ca_concavity, compute_ca_electrostatics,
)
from sasa import compute_dssp_sasa
from struct_dataset import _parse_positions, binding_site_collate_fn


def extract_full_features(pdb_result):
    """全长逐残基结构特征 (不截断)。ca_exposure 占位, 由真 DSSP SASA 填充。"""
    N_coords = pdb_result['N']
    CA_coords = pdb_result['CA']
    C_coords = pdb_result['C']
    O_coords = pdb_result['O']
    sequence = pdb_result['sequence']
    L = pdb_result['L']

    aa_indices = np.array([AA_TO_IDX.get(aa, 20) for aa in sequence], dtype=np.int64)
    dihedral_angles = extract_backbone_dihedrals(N_coords, CA_coords, C_coords)
    dihedral_sincos = encode_dihedral_sincos(dihedral_angles, n_freqs=2)   # (L, 12)
    frames = build_backbone_frames(N_coords, CA_coords, C_coords)
    backbone_geom = compute_backbone_bond_geometry(N_coords, CA_coords, C_coords)
    local_coords = convert_to_local_coordinates(N_coords, CA_coords, C_coords, O_coords, frames)
    # 表面特征按全长算 (全局口袋/电荷环境), 不按窗口
    ca_concavity = compute_ca_concavity(CA_coords)
    ca_electrostatics = compute_ca_electrostatics(CA_coords, sequence)

    return {
        'aa_indices': aa_indices,
        'dihedral_angles': dihedral_angles,
        'dihedral_sincos': dihedral_sincos,
        'backbone_frames_R': frames['R'],
        'backbone_frames_t': frames['t'],
        'backbone_frames_quat': frames['quat'],
        'backbone_geom': backbone_geom,
        'local_atom_coords': local_coords,
        'ca_exposure': np.zeros(L, dtype=np.float32),   # 下面用真 DSSP 填充
        'ca_concavity': ca_concavity,
        'ca_electrostatics': ca_electrostatics,
        'mask': np.ones(L, dtype=np.float32),
        'sequence': sequence,
    }


def load_full_protein(pdb_dir, pdb_id, dna_positions, rna_positions):
    """加载一条蛋白的全长特征 + 全长逐残基标签 (numpy dict)。

    Returns:
        full: dict, 含 extract_full_features 的所有键 + 'dna_label'/'rna_label' (L,)
    """
    pdb_path = None
    for ext in ['.pdb', '.ent']:
        cand = os.path.join(pdb_dir, f'{pdb_id}{ext}')
        if os.path.exists(cand):
            pdb_path = cand
            break
    if pdb_path is None:
        raise FileNotFoundError(f'PDB not found: {pdb_id}')

    pdb_result = parse_pdb_backbone(pdb_path)
    full = extract_full_features(pdb_result)
    L = pdb_result['L']

    dna_label = np.zeros(L, dtype=np.float32)
    rna_label = np.zeros(L, dtype=np.float32)
    for pos in _parse_positions(dna_positions):
        if 0 <= pos < L:
            dna_label[pos] = 1.0
    for pos in _parse_positions(rna_positions):
        if 0 <= pos < L:
            rna_label[pos] = 1.0

    # 真 DSSP 全长 SASA (相对可及性 [0,1]); 失败则保持 0 占位
    try:
        _, dssp_rsa, _ = compute_dssp_sasa(pdb_path)
        if len(dssp_rsa) >= L:
            full['ca_exposure'] = dssp_rsa[:L].astype(np.float32)
    except Exception:
        pass

    full['dna_label'] = dna_label
    full['rna_label'] = rna_label
    return full


def _slice_window(full, s, L):
    """把全长 numpy 特征 dict 切成 [s:s+L] 的窗口。"""
    sl = slice(s, s + L)
    return {
        'aa_indices': full['aa_indices'][sl],
        'dihedral_angles': full['dihedral_angles'][sl],
        'dihedral_sincos': full['dihedral_sincos'][sl],
        'backbone_frames_R': full['backbone_frames_R'][sl],
        'backbone_frames_t': full['backbone_frames_t'][sl],
        'backbone_frames_quat': full['backbone_frames_quat'][sl],
        'backbone_geom': full['backbone_geom'][sl],
        'local_atom_coords': full['local_atom_coords'][sl],
        'ca_exposure': full['ca_exposure'][sl],
        'ca_concavity': full['ca_concavity'][sl],
        'ca_electrostatics': full['ca_electrostatics'][sl],
        'mask': full['mask'][sl],
        'sequence': full['sequence'][sl],
    }


class RandomWindowDataset(Dataset):
    """每次 __getitem__ 从全长蛋白随机抽一个 window 长度的窗口。

    pos_bias: 正样本偏置采样概率 [0,1]。结合位点是连续斑块(正残基聚成簇), 纯均匀
    采样对长蛋白会整窗错过斑块, 导致训练窗口大量"全负"、正例欠采样。以 pos_bias
    概率把窗口中心对准一个结合残基(dna 或 rna), 其余概率均匀采样保留负样本。
    """

    def __init__(self, pdb_dir, label_df, window=512, pos_bias=0.0):
        self.pdb_dir = pdb_dir
        self.window = window
        self.pos_bias = pos_bias
        self.df = label_df.reset_index(drop=True)
        # pdb_id -> (dna_positions_str, rna_positions_str)
        self._label_map = {}
        for _, r in self.df.iterrows():
            pid = str(r['pdb_id']).strip()
            self._label_map[pid] = (r.get('dna_positions', ''), r.get('rna_positions', ''))
        self._full = {}   # 每 worker 独立的全长缓存

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pdb_id = str(row['pdb_id']).strip()
        full = self._full.get(pdb_id)
        if full is None:
            dna_pos, rna_pos = self._label_map[pdb_id]
            full = load_full_protein(self.pdb_dir, pdb_id, dna_pos, rna_pos)
            self._full[pdb_id] = full

        L_full = full['aa_indices'].shape[0]
        W = self.window
        if L_full <= W:
            s, L = 0, L_full
        else:
            pos = np.where((full['dna_label'] + full['rna_label']) > 0.5)[0]
            if self.pos_bias > 0 and len(pos) > 0 and np.random.rand() < self.pos_bias:
                c = int(np.random.choice(pos))
                s = int(np.clip(c - W // 2, 0, L_full - W))
            else:
                s = np.random.randint(0, L_full - W + 1)
            L = W

        win = _slice_window(full, s, L)
        return {
            'aa_indices': torch.tensor(win['aa_indices'], dtype=torch.long),
            'dihedral_angles': torch.tensor(win['dihedral_angles'], dtype=torch.float32),
            'dihedral_sincos': torch.tensor(win['dihedral_sincos'], dtype=torch.float32),
            'backbone_frames_R': torch.tensor(win['backbone_frames_R'], dtype=torch.float32),
            'backbone_frames_t': torch.tensor(win['backbone_frames_t'], dtype=torch.float32),
            'backbone_frames_quat': torch.tensor(win['backbone_frames_quat'], dtype=torch.float32),
            'backbone_geom': torch.tensor(win['backbone_geom'], dtype=torch.float32),
            'local_atom_coords': torch.tensor(win['local_atom_coords'], dtype=torch.float32),
            'ca_exposure': torch.tensor(win['ca_exposure'], dtype=torch.float32),
            'ca_concavity': torch.tensor(win['ca_concavity'], dtype=torch.float32),
            'ca_electrostatics': torch.tensor(win['ca_electrostatics'], dtype=torch.float32),
            'mask': torch.tensor(win['mask'], dtype=torch.float32),
            'dna_label': torch.tensor(full['dna_label'][s:s + L], dtype=torch.float32),
            'rna_label': torch.tensor(full['rna_label'][s:s + L], dtype=torch.float32),
            'pdb_id': pdb_id,
            'sequence': win['sequence'],
            'L': L,
        }
