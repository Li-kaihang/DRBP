#!/usr/bin/env python3
"""Binding site 预测的数据集与 collate。

复用 data_processing 的 parse_pdb_backbone + extract_structure_features 提取结构特征,
额外加入逐残基的 DNA/RNA 结合标签。
"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data_processing import parse_pdb_backbone, extract_structure_features
from sasa import compute_dssp_sasa


def _parse_positions(s):
    """'0,5,10' -> [0, 5, 10]; 空串 -> []"""
    if not isinstance(s, str) or not s.strip():
        return []
    return [int(x) for x in s.split(',') if x.strip() != '']


class BindingSiteDataset(Dataset):
    """从 PDB 结构 + binding_site_labels.csv 加载逐残基 DNA/RNA 结合标签。"""

    def __init__(self, pdb_dir: str, label_df, max_len: int = 512):
        """Args:
            pdb_dir:  存放 .pdb 文件的目录
            label_df: pandas DataFrame, 必须有列 pdb_id, sequence, dna_positions, rna_positions
            max_len:  最大序列长度
        """
        self.pdb_dir = pdb_dir
        self.max_len = max_len
        self.df = label_df.reset_index(drop=True)
        self._cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pdb_id = str(row['pdb_id']).lower().strip()
        if pdb_id in self._cache:
            return self._cache[pdb_id]

        pdb_path = None
        for ext in ['.pdb', '.ent']:
            cand = os.path.join(self.pdb_dir, f'{pdb_id}{ext}')
            if os.path.exists(cand):
                pdb_path = cand
                break
        if pdb_path is None:
            raise FileNotFoundError(f'PDB not found: {pdb_id}')

        pdb_result = parse_pdb_backbone(pdb_path)
        features = extract_structure_features(pdb_result, self.max_len, include_pair=False)
        L = features['aa_indices'].shape[0]

        # 用真实 DSSP SASA 替换 Cα 级代理 (ca_exposure)。DSSP 全原子 + 水探针, 更准且 reviewer 认可。
        try:
            _, dssp_rsa, _ = compute_dssp_sasa(pdb_path)
            if len(dssp_rsa) >= L:
                features['ca_exposure'] = dssp_rsa[:L].astype(np.float32)
        except Exception:
            pass   # DSSP 失败则回退到 Cα 级代理

        # 逐残基标签 (与结构特征/序列同序, 0-indexed)
        dna_label = torch.zeros(L, dtype=torch.float32)
        rna_label = torch.zeros(L, dtype=torch.float32)
        for pos in _parse_positions(row.get('dna_positions', '')):
            if 0 <= pos < L:
                dna_label[pos] = 1.0
        for pos in _parse_positions(row.get('rna_positions', '')):
            if 0 <= pos < L:
                rna_label[pos] = 1.0

        result = {
            'aa_indices': torch.tensor(features['aa_indices'], dtype=torch.long),
            'dihedral_angles': torch.tensor(features['dihedral_angles'], dtype=torch.float32),
            'dihedral_sincos': torch.tensor(features['dihedral_sincos'], dtype=torch.float32),
            'backbone_frames_R': torch.tensor(features['backbone_frames_R'], dtype=torch.float32),
            'backbone_frames_t': torch.tensor(features['backbone_frames_t'], dtype=torch.float32),
            'backbone_frames_quat': torch.tensor(features['backbone_frames_quat'], dtype=torch.float32),
            'backbone_geom': torch.tensor(features['backbone_geom'], dtype=torch.float32),
            'local_atom_coords': torch.tensor(features['local_atom_coords'], dtype=torch.float32),
            'ca_exposure': torch.tensor(features['ca_exposure'], dtype=torch.float32),
            'ca_concavity': torch.tensor(features['ca_concavity'], dtype=torch.float32),
            'ca_electrostatics': torch.tensor(features['ca_electrostatics'], dtype=torch.float32),
            'mask': torch.tensor(features['mask'], dtype=torch.float32),
            'dna_label': dna_label,
            'rna_label': rna_label,
            'pdb_id': pdb_id,
            'sequence': features['sequence'],
            'L': L,
        }
        self._cache[pdb_id] = result
        return result


def binding_site_collate_fn(batch):
    """结构 batch padding + 逐残基标签 padding。"""
    batch_max_len = max(item['L'] for item in batch)
    B = len(batch)

    def _zeros(*shape, dtype=torch.float32):
        return torch.zeros(*shape, dtype=dtype)

    aa_indices = _zeros(B, batch_max_len, dtype=torch.long)
    dihedral_angles = _zeros(B, batch_max_len, 3)
    dihedral_sincos = _zeros(B, batch_max_len, batch[0]['dihedral_sincos'].shape[-1])
    backbone_frames_R = _zeros(B, batch_max_len, 3, 3)
    backbone_frames_t = _zeros(B, batch_max_len, 3)
    backbone_frames_quat = _zeros(B, batch_max_len, 4)
    backbone_geom = _zeros(B, batch_max_len, batch[0]['backbone_geom'].shape[-1])
    local_atom_coords = _zeros(B, batch_max_len, batch[0]['local_atom_coords'].shape[-1])
    ca_exposure = _zeros(B, batch_max_len)
    ca_concavity = _zeros(B, batch_max_len)
    ca_electrostatics = _zeros(B, batch_max_len)
    mask = _zeros(B, batch_max_len)
    dna_labels = _zeros(B, batch_max_len)
    rna_labels = _zeros(B, batch_max_len)
    pdb_ids, sequences = [], []

    for i, item in enumerate(batch):
        L = item['L']
        aa_indices[i, :L] = item['aa_indices']
        dihedral_angles[i, :L] = item['dihedral_angles']
        dihedral_sincos[i, :L] = item['dihedral_sincos']
        backbone_frames_R[i, :L] = item['backbone_frames_R']
        backbone_frames_t[i, :L] = item['backbone_frames_t']
        backbone_frames_quat[i, :L] = item['backbone_frames_quat']
        backbone_geom[i, :L] = item['backbone_geom']
        local_atom_coords[i, :L] = item['local_atom_coords']
        ca_exposure[i, :L] = item['ca_exposure']
        ca_concavity[i, :L] = item['ca_concavity']
        ca_electrostatics[i, :L] = item['ca_electrostatics']
        mask[i, :L] = 1.0
        dna_labels[i, :L] = item['dna_label']
        rna_labels[i, :L] = item['rna_label']
        pdb_ids.append(item['pdb_id'])
        sequences.append(item['sequence'])

    return {
        'aa_indices': aa_indices,
        'dihedral_angles': dihedral_angles,
        'dihedral_sincos': dihedral_sincos,
        'backbone_frames_R': backbone_frames_R,
        'backbone_frames_t': backbone_frames_t,
        'backbone_frames_quat': backbone_frames_quat,
        'backbone_geom': backbone_geom,
        'local_atom_coords': local_atom_coords,
        'ca_exposure': ca_exposure,
        'ca_concavity': ca_concavity,
        'ca_electrostatics': ca_electrostatics,
        'mask': mask,
        'dna_label': dna_labels,
        'rna_label': rna_labels,
        'pdb_id': pdb_ids,
        'sequence': sequences,
    }
