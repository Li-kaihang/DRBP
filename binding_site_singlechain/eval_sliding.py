#!/usr/bin/env python3
"""滑动窗口推理 + 标准测试集 (DNA-129 / RNA-117 / DNA-181) 全长评估。

对每条测试蛋白: window=512、stride=256 (50% 重叠) 滑过全长, 三角(tent)中心加权
拼接 → 全长逐残基 logits。按三个来源 (DNA-129 / DNA-181 / RNA-117) 分别 + 汇总
输出 AUROC / PR-AUC / MCC / F1 / precision / recall / 最优阈值。

DNA 指标在 DBP 蛋白上算, RNA 指标在 RBP 蛋白上算。测试 PDB 在 data/test_pdbs/,
pdb_id 保留大小写 (如 5dy0_A)。

用法: cd /root/DRBP/finetune_6layers/binding_site_data_v2 && \
      /root/.conda/envs/drbp/bin/python eval_sliding.py --ckpt best_bsite.pt \
      2>&1 | tee eval.log
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, precision_score, recall_score, f1_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from struct_gnn import StructureGNN
from joint_model import JointBindingSiteModel
from lora import apply_lora_to_esm
from randwin_dataset import load_full_protein

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_LABELS = os.path.join(HERE, 'data', 'test_labels.csv')
TEST_PDB_DIR = os.path.join(HERE, 'data', 'test_pdbs')
CKPT = os.path.join(HERE, 'best_bsite.pt')


def load_esm(device, model_name='facebook/esm2_t30_150M_UR50D'):
    from transformers import EsmTokenizer, EsmModel
    tok = EsmTokenizer.from_pretrained(model_name, local_files_only=True)
    esm = EsmModel.from_pretrained(model_name, local_files_only=True)
    esm.to(device)
    return esm, tok


def window_to_batch(full, s, L):
    sl = slice(s, s + L)
    return {
        'aa_indices': torch.tensor(full['aa_indices'][sl], dtype=torch.long).unsqueeze(0),
        'dihedral_angles': torch.tensor(full['dihedral_angles'][sl], dtype=torch.float32).unsqueeze(0),
        'dihedral_sincos': torch.tensor(full['dihedral_sincos'][sl], dtype=torch.float32).unsqueeze(0),
        'backbone_frames_R': torch.tensor(full['backbone_frames_R'][sl], dtype=torch.float32).unsqueeze(0),
        'backbone_frames_t': torch.tensor(full['backbone_frames_t'][sl], dtype=torch.float32).unsqueeze(0),
        'backbone_frames_quat': torch.tensor(full['backbone_frames_quat'][sl], dtype=torch.float32).unsqueeze(0),
        'backbone_geom': torch.tensor(full['backbone_geom'][sl], dtype=torch.float32).unsqueeze(0),
        'local_atom_coords': torch.tensor(full['local_atom_coords'][sl], dtype=torch.float32).unsqueeze(0),
        'ca_exposure': torch.tensor(full['ca_exposure'][sl], dtype=torch.float32).unsqueeze(0),
        'ca_concavity': torch.tensor(full['ca_concavity'][sl], dtype=torch.float32).unsqueeze(0),
        'ca_electrostatics': torch.tensor(full['ca_electrostatics'][sl], dtype=torch.float32).unsqueeze(0),
        'mask': torch.tensor(full['mask'][sl], dtype=torch.float32).unsqueeze(0),
    }


@torch.no_grad()
def sliding_predict(model, tok, device, full, window, stride):
    L_full = full['aa_indices'].shape[0]
    W = window
    if L_full <= W:
        starts = [0]
    else:
        starts = list(range(0, L_full - W + 1, stride))
        if starts[-1] != L_full - W:
            starts.append(L_full - W)

    sum_dna = np.zeros(L_full, dtype=np.float32)
    sum_rna = np.zeros(L_full, dtype=np.float32)
    sum_w = np.zeros(L_full, dtype=np.float32)

    for s in starts:
        L = min(W, L_full - s)
        batch = window_to_batch(full, s, L)
        seq = full['sequence'][s:s + L]
        enc = tok([seq], padding=True, truncation=True, max_length=W + 2, return_tensors='pt')
        iid = enc['input_ids'].to(device)
        am = enc['attention_mask'].to(device)
        sb = {k: v.to(device) for k, v in batch.items()}
        dl, rl = model(sb, iid, am)
        dl = dl[0, :L].float().cpu().numpy()
        rl = rl[0, :L].float().cpu().numpy()
        j = np.arange(L, dtype=np.float32)
        w = np.clip(np.minimum(j + 1, L - j) / (L / 2.0), 0.0, 1.0)
        sum_dna[s:s + L] += w * dl
        sum_rna[s:s + L] += w * rl
        sum_w[s:s + L] += w

    dna = sum_dna / np.maximum(sum_w, 1e-8)
    rna = sum_rna / np.maximum(sum_w, 1e-8)
    return dna, rna


def metrics(y, p):
    prob = 1.0 / (1.0 + np.exp(-p))
    auroc = roc_auc_score(y, prob)
    prauc = average_precision_score(y, prob)
    best_mcc, best_t = -1.0, 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        pred = (prob >= t).astype(int)
        m = matthews_corrcoef(y, pred)
        if m > best_mcc:
            best_mcc, best_t = m, t
    pred = (prob >= best_t).astype(int)
    return dict(AUROC=auroc, PR_AUC=prauc, MCC=best_mcc, F1=f1_score(y, pred),
                precision=precision_score(y, pred), recall=recall_score(y, pred), thr=best_t)


def haspos(s):
    return isinstance(s, str) and bool(s.strip())


def fmt(m):
    return (f"AUROC={m['AUROC']:.4f}  PR-AUC={m['PR_AUC']:.4f}  MCC={m['MCC']:.4f}  "
            f"F1={m['F1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  thr={m['thr']:.2f}")


_TEST_CACHE = {}


def _load_test_full(pid, dna_pos, rna_pos):
    if pid not in _TEST_CACHE:
        _TEST_CACHE[pid] = load_full_protein(TEST_PDB_DIR, pid, dna_pos, rna_pos)
    return _TEST_CACHE[pid]


def evaluate_on_test(model, tok, device, window=512, stride=256):
    """在三个标准测试集上评估 (model 需已在 device 上), 返回 {group: metrics}。"""
    model.eval()
    test = pd.read_csv(TEST_LABELS, dtype={'pdb_id': str})
    test['pdb_id'] = test['pdb_id'].astype(str).str.strip()

    pred_cache = {}
    for _, row in test.iterrows():
        pid = row['pdb_id']
        full = _load_test_full(pid, row.get('dna_positions', ''), row.get('rna_positions', ''))
        dna, rna = sliding_predict(model, tok, device, full, window, stride)
        pred_cache[pid] = (dna, rna, full['dna_label'], full['rna_label'])

    groups = [('DNA-129', 'dna'), ('DNA-181', 'dna'), ('RNA-117', 'rna'),
              ('DNA 总体 (129+181)', 'dna'), ('RNA 总体 (117)', 'rna')]
    results = {}
    for name, which in groups:
        if '总体' in name:
            sub = test[test.source.isin(['DNA-129', 'DNA-181'])] if which == 'dna' \
                else test[test.source == 'RNA-117']
        else:
            sub = test[test.source == name]
        sub = sub.reset_index(drop=True)
        all_p, all_y = [], []
        for _, row in sub.iterrows():
            pid = row['pdb_id']
            dna, rna, dl, rl = pred_cache[pid]
            if which == 'dna':
                if not haspos(row.get('dna_positions', '')):
                    continue
                all_p.append(dna); all_y.append(dl)
            else:
                if not haspos(row.get('rna_positions', '')):
                    continue
                all_p.append(rna); all_y.append(rl)
        if not all_p:
            results[name] = None
            continue
        p = np.concatenate(all_p); y = np.concatenate(all_y)
        m = metrics(y, p)
        m['n_res'] = len(y)
        m['n_pos'] = int(y.sum())
        results[name] = m
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--window', type=int, default=512)
    ap.add_argument('--stride', type=int, default=256)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seq_only_dna', action='store_true',
                    help='DNA 头只吃序列(ESM-2), RNA 头吃序列+结构')
    ap.add_argument('--context_head', action='store_true',
                    help='融合特征后加残基间 self-attention (与训练一致)')
    ap.add_argument('--gated_fusion', action='store_true',
                    help='门控融合 (与训练一致, 否则 checkpoint 加载不上)')
    ap.add_argument('--lora_r', type=int, default=0,
                    help='LoRA 秩 r (与训练一致, 否则 checkpoint 加载不上)')
    ap.add_argument('--lora_alpha', type=int, default=32,
                    help='LoRA 缩放 alpha (与训练一致)')
    ap.add_argument('--esm_model', default='facebook/esm2_t30_150M_UR50D',
                    help='ESM-2 模型名或本地路径 (650M: /root/DRBP/models/esm2_t33_650M_UR50D)')
    ap.add_argument('--dropout', type=float, default=0.0,
                    help='fusion/两个头 的 dropout 概率 (训练时用的值, 仅保证结构一致, eval 时关闭)')
    a = ap.parse_args()
    device = a.device if torch.cuda.is_available() else 'cpu'

    esm, tok = load_esm(device, a.esm_model)
    if getattr(a, 'lora_r', 0) > 0:
        # n_layers=6 与 train 默认 unfreeze_esm_layers=6 一致
        apply_lora_to_esm(esm, r=a.lora_r, alpha=a.lora_alpha, dropout=0.1, n_layers=6)
    d_seq = esm.config.hidden_size
    gnn = StructureGNN(node_dim=256, edge_dim=64, n_layers=4, k_neighbors=16)
    model = JointBindingSiteModel(esm, gnn, d_seq=d_seq, d_struct=256,
                                  seq_only_dna=a.seq_only_dna,
                                  dropout=a.dropout,
                                  context_head=a.context_head,
                                  gated_fusion=a.gated_fusion).to(device)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    print(f"加载 {a.ckpt}: epoch {ck.get('epoch')}", flush=True)

    results = evaluate_on_test(model, tok, device, a.window, a.stride)

    print('\n=== 标准测试集结果 ===', flush=True)
    for name in ['DNA-129', 'DNA-181', 'RNA-117', 'DNA 总体 (129+181)', 'RNA 总体 (117)']:
        m = results.get(name)
        if m is None:
            print(f'\n{name}: 无阳性蛋白，跳过', flush=True)
            continue
        print(f'\n{name}: 残基 {m["n_res"]}（阳性 {m["n_pos"]}）', flush=True)
        print(f'  {fmt(m)}', flush=True)

    print('\n=== 参考 (文献，口径不同仅供方向参考) ===', flush=True)
    print('  GraphBind 原文: DNA-129 MCC 0.499 / RNA-117 MCC 0.322', flush=True)
    print('  ESM-NBR (我们实测 BioLiP): DNA 0.506 / RNA 0.215', flush=True)


if __name__ == '__main__':
    main()
