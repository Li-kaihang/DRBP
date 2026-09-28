#!/usr/bin/env python3
"""
DRBP 多标签联合学习 —— 训练脚本
================================
  /root/.conda/envs/drbp/bin/python train_drbp.py --n_gpus 4 --mode esm2
  /root/.conda/envs/drbp/bin/python train_drbp.py --n_gpus 4 --mode joint

相比 train_baseline.py 的改动:
  * 三个头 = DBP / RBP / DRBP (旧版第三个是 non, 无独立信息)
  * 联合损失: pos_weight + DRBP 直接监督 + 层级一致性
  * 测试集单卡普通 DataLoader —— 旧版对测试集也用 DistributedSampler, 4 卡下
    206 被 padding 到 208 (复制开头 2 条), 指标是在含重复样本的数据上算的
  * val 上按 MCC 搜阈值, 测试时同时报搜到的阈值和固定 0.5 两组
  * 每次 run 独立目录 runs/{ts}_{tag}/, 不再互相覆盖
  * 动态 padding (按 batch 内最长), 平均蛋白 387 残基, 固定 pad 到 1026 浪费 2.6x
"""

import os
import sys
import csv
import json
import time
import socket
import argparse
import contextlib
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split

from model_baseline import Config
from model_drbp import DRBPNetNew, drbp_loss, non_loss
from metrics import (compute_metrics, collapse_diagnostics, search_thresholds,
                     CLASS_NAMES, drbp_decision, macro_f1)
import struct_adapter as SA

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "parsed")
RUNS_DIR = os.path.join(HERE, "runs")


# ============================================================
# DDP 工具 (沿用 train_baseline.py, 这部分逻辑本身没问题)
# ============================================================

def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def ddp_setup():
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        return local_rank, dist.get_world_size(), dist.get_rank()
    return 0, 1, 0


def is_main(rank):
    return rank == 0


def gather_1d(t, world_size, device):
    """跨卡收集 1D 数组 (各卡长度可能不同)"""
    x = torch.from_numpy(np.ascontiguousarray(t)).float().to(device)
    n = torch.tensor([x.shape[0]], dtype=torch.long, device=device)
    sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(sizes, n)
    m = max(s.item() for s in sizes)
    if x.shape[0] < m:
        x = torch.cat([x, torch.zeros(m - x.shape[0], device=device)])
    buf = [torch.zeros(m, device=device) for _ in range(world_size)]
    dist.all_gather(buf, x)
    return torch.cat([b[:s.item()] for b, s in zip(buf, sizes)]).cpu().numpy()


# ============================================================
# 数据集
# ============================================================

class DRBPDataset(Dataset):
    """
    joint 模式下同时产出结构特征。结构缺失不报错、不过滤 —— 产出空结构, GNN 贡献
    被 mask 乘成精确的 0, joint 自动退化成纯序列。这样两种模式跑的是同一批样本,
    消融才公平 (EZL 只有 79.5% 有结构, 过滤会让两边测试集对不上)。
    """

    def __init__(self, df, tokenizer, max_len=1024, use_struct=False):
        self.ids = df['protein_id'].astype(str).tolist()
        self.seqs = df['seq'].tolist()
        self.dbp = df['DBP_label'].astype(np.float32).values
        self.rbp = df['RBP_label'].astype(np.float32).values
        self.tok = tokenizer
        self.max_len = max_len
        self.use_struct = use_struct

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        seq = self.seqs[i][:self.max_len]
        # 不 padding, 由 collate 按 batch 内最长补齐
        enc = self.tok(seq, truncation=True, max_length=self.max_len + 2,
                       return_tensors=None)
        item = {
            'input_ids': torch.tensor(enc['input_ids'], dtype=torch.long),
            'n_res': len(seq),
            'dbp': self.dbp[i],
            'rbp': self.rbp[i],
        }
        if self.use_struct:
            feat = SA.load_or_build(self.ids[i], self.max_len, build=False)
            if feat is not None:
                r2s, n = SA.build_res2struct(feat['sequence'], self.seqs[i], self.max_len)
                if n == 0:
                    feat, r2s = None, None
            else:
                r2s = None
            item['struct'] = feat
            item['r2s'] = r2s
        return item


def make_collate(pad_id, use_struct):
    def collate(batch):
        Lt = max(x['input_ids'].shape[0] for x in batch)
        Lr = Lt - 2                                   # 去掉 CLS/EOS
        B = len(batch)
        ids = torch.full((B, Lt), pad_id, dtype=torch.long)
        am = torch.zeros(B, Lt, dtype=torch.long)
        rm = torch.zeros(B, Lr)
        for i, x in enumerate(batch):
            n = x['input_ids'].shape[0]
            ids[i, :n] = x['input_ids']
            am[i, :n] = 1
            rm[i, :min(x['n_res'], Lr)] = 1.0         # 只盖真实残基, 不含 EOS
        out = {
            'input_ids': ids,
            'attention_mask': am,
            'residue_mask': rm,
            'dbp': torch.tensor([x['dbp'] for x in batch]),
            'rbp': torch.tensor([x['rbp'] for x in batch]),
        }
        if use_struct:
            sb, r2s = SA.collate_struct([(x['struct'], x['r2s']) for x in batch], Lr)
            out['struct_batch'] = sb
            out['res2struct'] = r2s
        return out
    return collate


def move(batch, device):
    out = {}
    for k, v in batch.items():
        if k == 'struct_batch':
            out[k] = {kk: vv.to(device) for kk, vv in v.items()}
        else:
            out[k] = v.to(device)
    return out


# ============================================================
# 前向 / 评估
# ============================================================

def run_model(model, b):
    return model(b['input_ids'], b['attention_mask'], b['residue_mask'],
                 b.get('struct_batch'), b.get('res2struct'))


@torch.no_grad()
def collect_predictions(model, loader, device, world_size):
    model.eval()
    P = {k: [] for k in ['d', 'r', 'b', 'yd', 'yr']}
    for batch in loader:
        b = move(batch, device)
        o = run_model(model, b)
        P['d'].append(o['dbp_prob'].float().cpu().numpy())
        P['r'].append(o['rbp_prob'].float().cpu().numpy())
        P['b'].append(o['drbp_prob'].float().cpu().numpy())
        P['yd'].append(b['dbp'].cpu().numpy())
        P['yr'].append(b['rbp'].cpu().numpy())
    P = {k: np.concatenate(v) if v else np.zeros(0) for k, v in P.items()}
    if world_size > 1:
        P = {k: gather_1d(v, world_size, device) for k, v in P.items()}
        n = len(loader.dataset)
        P = {k: v[:n] for k, v in P.items()}
    return P['d'], P['r'], P['b'], P['yd'], P['yr']


@torch.no_grad()
def eval_loss(model, loader, device, world_size, W, args):
    model.eval()
    tot, nb = 0.0, 0
    for batch in loader:
        b = move(batch, device)
        o = run_model(model, b)
        if args.head_scheme == 'non':
            l, _ = non_loss(o, b['dbp'], b['rbp'], W['d'], W['r'], W['n'])
        else:
            l, _ = drbp_loss(o, b['dbp'], b['rbp'], W['d'], W['r'], W['b'],
                             args.lam1, args.lam2, args.cons_mode,
                             args.w_cons_pos, args.lam3, args.conj_margin)
        tot += l.item()
        nb += 1
    t = torch.tensor([tot, nb], device=device)
    if world_size > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t[0] / max(t[1].item(), 1)).item()


# ============================================================
# 日志
# ============================================================

def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       cwd=HERE, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def data_stats(df, name):
    n = len(df)
    pd_, pr = (df.DBP_label == 1).mean(), (df.RBP_label == 1).mean()
    pb = ((df.DBP_label == 1) & (df.RBP_label == 1)).mean()
    return {
        f'{name}_n': n,
        f'{name}_non': int(((df.DBP_label == 0) & (df.RBP_label == 0)).sum()),
        f'{name}_dbp_only': int(((df.DBP_label == 1) & (df.RBP_label == 0)).sum()),
        f'{name}_rbp_only': int(((df.DBP_label == 0) & (df.RBP_label == 1)).sum()),
        f'{name}_drbp': int((pb * n).round()),
        f'{name}_P_dbp': round(float(pd_), 4),
        f'{name}_P_rbp': round(float(pr), 4),
        f'{name}_P_both': round(float(pb), 4),
        # lift<1 说明 DBP/RBP 负相关, 这是 DRBP 难学的根源
        f'{name}_lift': round(float(pb / (pd_ * pr)), 4) if pd_ * pr > 0 else float('nan'),
    }


def struct_coverage(df):
    hit = sum(1 for p in df['protein_id'].astype(str)
              if os.path.exists(SA.cache_path(SA.resolve_structure(p)[2])))
    return hit, len(df)


def write_config(run_dir, args, extra):
    rows = {**vars(args), **extra,
            'git_commit': git_commit(),
            'cmdline': ' '.join(sys.argv),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    with open(os.path.join(run_dir, 'config.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['key', 'value'])
        for k, v in rows.items():
            w.writerow([k, v])


def write_arch(run_dir, model):
    """供事后查模型细节: 完整结构 + 每模块参数量"""
    with open(os.path.join(run_dir, 'model_arch.txt'), 'w') as f:
        f.write(str(model) + '\n\n')
        f.write(f"{'module':<48}{'params':>14}{'trainable':>12}\n")
        f.write('-' * 74 + '\n')
        for name, mod in model.named_children():
            tot = sum(p.numel() for p in mod.parameters())
            tr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            f.write(f"{name:<48}{tot:>14,}{tr:>12,}\n")
        tot = sum(p.numel() for p in model.parameters())
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write('-' * 74 + '\n')
        f.write(f"{'TOTAL':<48}{tot:>14,}{tr:>12,}\n")


def evaluate_test_sets(raw, tokenizer, collate, args, device, use_struct,
                       best_t, run_dir, epoch, dump_predictions=True):
    """
    在 4 个测试集上评估。只在 rank0 调用, 用普通 DataLoader (不是 DistributedSampler)
    —— 旧版对测试集用 DistributedSampler, 4 卡下 206 被 padding 到 208, 指标是在含
    重复样本的数据上算的。

    返回 rows(list[dict]), 每个测试集两行: 搜到的阈值 / 固定 0.5。
    """
    raw.eval()
    rows = []
    for name in args.test_sets:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))
        ds = DRBPDataset(df, tokenizer, args.max_len, use_struct)
        ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=args.workers)
        dp, rp, bp, dy, ry = collect_predictions(raw, ld, device, 1)
        cov = struct_coverage(df) if use_struct else (0, len(df))

        if dump_predictions:
            # 逐蛋白明细, 供 case 分析 (哪些 DRBP 被漏了 / 哪些 non 被误报)
            t_d, t_r, t_b = best_t
            pred_cls = np.zeros(len(df), int)
            pred_cls[(dp > t_d) & (rp <= t_r)] = 1
            pred_cls[(dp <= t_d) & (rp > t_r)] = 2
            pred_cls[(dp > t_d) & (rp > t_r)] = 3
            pred_cls[drbp_decision(dp, rp, bp, t_d, t_r, t_b,
                                   use_drbp_head=(args.head_scheme != 'non'))] = 3
            true_cls = np.zeros(len(df), int)
            true_cls[(dy == 1) & (ry == 0)] = 1
            true_cls[(dy == 0) & (ry == 1)] = 2
            true_cls[(dy == 1) & (ry == 1)] = 3
            has_st = [os.path.exists(SA.cache_path(SA.resolve_structure(p)[2]))
                      for p in df['protein_id'].astype(str)] if use_struct else [False] * len(df)
            pd.DataFrame({
                'protein_id': df['protein_id'].values,
                'seq_len': df['seq'].str.len().values,
                'has_struct': has_st,
                'true_dbp': dy.astype(int), 'true_rbp': ry.astype(int),
                'true_class': [CLASS_NAMES[c] for c in true_cls],
                'dbp_prob': dp.round(4), 'rbp_prob': rp.round(4), 'drbp_prob': bp.round(4),
                'pred_class': [CLASS_NAMES[c] for c in pred_cls],
                'correct': (pred_cls == true_cls).astype(int),
            }).to_csv(os.path.join(run_dir, f'predictions_{name}.csv'), index=False)

        for label, (t_d, t_r, t_b) in [('searched', best_t), ('fixed0.5', (0.5, 0.5, 0.5))]:
            m = compute_metrics(dp, rp, bp, dy, ry, t_d, t_r, t_b,
                                use_drbp_head=(args.head_scheme != 'non'))
            m.update(collapse_diagnostics(dp, rp, bp))
            # 每行自带 epoch 和关键超参, 这样单看一行就知道是什么配置跑出来的
            m.update({'epoch': epoch, 'test_set': name, 'thr': label, 'n': len(df),
                      'struct_cov': f'{cov[0]}/{cov[1]}',
                      'mode': args.mode, 'lam1': args.lam1, 'lam2': args.lam2,
                      'pw_cap': args.pw_cap, 'cons_mode': args.cons_mode,
                      'crosstalk': int(not args.no_crosstalk),
                      'lr': args.lr, 'batch_size': args.batch_size,
                      'd_model': args.d_model, 'dropout': args.dropout,
                      'train_csv': args.train_csv})
            rows.append(m)
    raw.train()
    return rows


TEST_HEAD_COLS = ['epoch', 'test_set', 'thr', 'n', 'struct_cov', 'mode']


def append_test_rows(run_dir, rows, fname='test_history.csv'):
    """追加写。第一次写表头, 之后只追加。"""
    if not rows:
        return
    path = os.path.join(run_dir, fname)
    cols = TEST_HEAD_COLS + [k for k in rows[0] if k not in TEST_HEAD_COLS]
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerows(rows)


def print_test_table(rows, title):
    print(f"\n--- {title} ---", flush=True)
    print(f"{'test_set':<10}{'thr':<11}{'dbp_auc':>9}{'rbp_auc':>9}{'drbp_auc':>10}"
          f"{'acc':>8}{'mcc':>8}{'DRBP捕获':>12}{'pred':>7}{'prec':>7}{'f1':>7}", flush=True)
    for m in rows:
        print(f"{m['test_set']:<10}{m['thr']:<11}{m['dbp_auc']:>9.4f}{m['rbp_auc']:>9.4f}"
              f"{m['drbp_auc']:>10.4f}{m['acc']:>8.4f}{m['mcc']:>8.4f}"
              f"{str(m['drbp_caught']) + '/' + str(m['drbp_total']):>12}"
              f"{m['drbp_pred']:>7}{m['drbp_prec']:>7.3f}{m['drbp_f1']:>7.3f}", flush=True)


METRIC_COLS = ['epoch', 'train_loss', 'val_loss', 'l_dbp', 'l_rbp', 'l_drbp', 'l_cons',
               'l_conj', 'l_non', 'coact', 'med_p_weak',
               'dbp_auc', 'rbp_auc', 'drbp_auc', 'acc', 'mcc',
               'drbp_caught', 'drbp_total', 'drbp_pred', 'drbp_prec', 'drbp_rec', 'drbp_f1',
               'corr_dr', 'corr_db', 'corr_rb',
               'frac_d_pos', 'frac_r_pos', 'frac_b_pos', 'std_d', 'std_r', 'std_b',
               'consistency_mse', 'pred_non', 'pred_dbp', 'pred_rbp', 'pred_drbp',
               'best_t_d', 'best_t_r', 'best_t_b', 'best_mcc',
               'val_macro_f1', 'macro_f1_srch', 'mean_auc', 'sel_raw', 'sel_ema',
               'alpha_d', 'alpha_r', 'lr', 'best']

SELECT_METRICS = ['composite', 'mean_auc', 'macro_f1', 'mcc', 'macro_f1_fix']


def selection_score(m, macro_f1_fixed, how):
    """
    选最优 checkpoint 的打分, 越大越好。返回 (score, mean_auc)。

    默认 composite = 0.5 × mean(三头 AUC) + 0.5 × macro-F1(搜索阈值下)。

    为什么不再用原来的 macro-F1@固定0.5:
      val 里 DRBP 只有 39 条, 硬判定指标在这个样本量上量化噪声极大 —— esm2 跑满 16
      个 epoch, drbp_f1 std=0.033 且末值比首值还低 (0.198→0.175), prec 在
      0.115~0.412 之间乱蹦; 而同样这 39 条算出来的 drbp_auc std=0.018 且趋势干净。
      噪声来自"硬判定 + 小样本", 不是样本量本身: AUC 用的是 39×2251 对两两比较, F1
      只用一个阈值切一刀。
      再叠上固定 0.5 的问题: 每个 epoch 搜出来的最优 t_b 在 0.10~0.95 之间漂
      (std=0.26), 所以 macro-F1@0.5 有一半在量"这个 epoch 的校准碰巧对没对上 0.5"。
      净结果: val_macro_f1 全程 std=0.030, 而 ep2→ep16 的真实漂移只有 +0.014 ——
      噪声是信号的两倍, 谁当 best 基本靠抽签 (joint 就是 ep2 抽到 0.7378 顶到天花板,
      之后 AUC 一路涨到 ep6 也再没进过 best)。

    为什么是两项相加而不是二选一:
      AUC 那半给趋势 (dbp/rbp std 只有 0.003/0.008), 但它只管排序 —— 某一类在硬判定
      上完全塌掉, AUC 照样很漂亮。macro-F1 那半守住"四类都没塌"的底线, 这正是原来
      那个指标的正确意图, 保留。
    """
    aucs = [a for a in (m['dbp_auc'], m['rbp_auc'], m['drbp_auc']) if a == a]   # 滤 nan
    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    score = {
        'composite':    0.5 * mean_auc + 0.5 * m['macro_f1'],
        'mean_auc':     mean_auc,
        'macro_f1':     m['macro_f1'],        # 搜索阈值下
        'mcc':          m['mcc'],
        'macro_f1_fix': macro_f1_fixed,       # 旧行为, 只为复现历史 run
    }[how]
    return float(score), mean_auc


# ============================================================
# 主流程
# ============================================================

def main_worker(local_rank, world_size, args):
    if world_size is not None and world_size > 1 and 'RANK' not in os.environ:
        os.environ['RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)

    local_rank, world_size, rank = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    use_struct = (args.mode in ('joint', 'struct'))

    # ---------------- 数据 ----------------
    train_df = pd.read_csv(os.path.join(DATA_DIR, args.train_csv))
    if args.limit:
        k = max(1, args.limit // 4)
        train_df = pd.concat(
            [g.sample(min(len(g), k), random_state=42)
             for _, g in train_df.groupby(['DBP_label', 'RBP_label'])],
            ignore_index=True)

    strat = (train_df['DBP_label'] * 2 + train_df['RBP_label']).astype(int)
    train_df, val_df = train_test_split(train_df, test_size=args.val_size,
                                        random_state=42, stratify=strat)

    # ---- DRBP 过采样 (数据层面别让 DRBP 吃亏) ----
    # 根因: DRBP 只占 1.7% (395/20602), DRBP 头是稀有类检测器 → 概率校准坏、阈值乱跳;
    # 同时两个主头被 98.3% 的"非此即彼"样本教会互斥 (真 DRBP 上同时开火只有 21/103)。
    # 只在训练集上过采样, 验证集保持原分布 (干净)。
    if args.drbp_upsample > 1:
        dr = train_df[(train_df['DBP_label'] == 1) & (train_df['RBP_label'] == 1)]
        reps = int(round(args.drbp_upsample)) - 1          # 额外复制的份数
        if len(dr) > 0 and reps > 0:
            train_df = pd.concat([train_df] + [dr] * reps, ignore_index=True)
        print(f"DRBP 过采样 ×{args.drbp_upsample:.1f}: train DRBP {len(dr)} → "
              f"{len(dr) * int(round(args.drbp_upsample))}, 占比 {len(dr)*args.drbp_upsample/len(train_df):.1%}",
              flush=True)

    # ---------------- 模型 ----------------
    from transformers import EsmTokenizer, EsmModel
    tokenizer = EsmTokenizer.from_pretrained(args.esm_model, local_files_only=True)
    # 纯结构模式只用结构 GNN, 不加载 ESM 序列编码器 (省 ~600MB 显存, 语义上不依赖序列)
    esm = None
    if args.mode != 'struct':
        esm = EsmModel.from_pretrained(args.esm_model, local_files_only=True)
        # 解冻 ESM 最后 N 层 (微调实验)。0 = 全冻结(默认)
        if args.unfreeze_esm_layers > 0:
            for p in esm.parameters():
                p.requires_grad = False
            n = len(esm.encoder.layer)
            for i in range(n - args.unfreeze_esm_layers, n):
                for p in esm.encoder.layer[i].parameters():
                    p.requires_grad = True
            print(f"解冻 ESM 最后 {args.unfreeze_esm_layers} 层 (共 {n} 层)", flush=True)

    gnn = None
    if use_struct:
        sys.path.insert(0, '/root/DRBP/final/classification')
        from model import StructureGNN
        gnn = StructureGNN(node_dim=args.gnn_dim, n_layers=args.gnn_layers,
                           k_neighbors=args.gnn_k)

    # esm_embed_dim 从实际加载的 ESM 读 (150M=640 / 650M=1280)，不再写死在 Config 里。
    esm_embed = esm.config.hidden_size if esm is not None else 640
    cfg = Config(esm_model_name=args.esm_model, esm_embed_dim=esm_embed,
                 freeze_esm=(not args.unfreeze_esm) and (args.unfreeze_esm_layers == 0),
                 d_model=args.d_model, n_heads=args.n_heads, dropout=args.dropout,
                 max_seq_len=args.max_len)
    cfg.mode = args.mode
    cfg.gnn_dim = args.gnn_dim
    cfg.head_hidden = args.head_hidden
    cfg.use_crosstalk = not args.no_crosstalk
    cfg.use_cross_label = not args.no_cross_label
    cfg.cla_heads = args.cla_heads
    cfg.head_scheme = args.head_scheme
    cfg.use_gate = not args.no_gate

    model = DRBPNetNew(cfg, esm, gnn).to(device)
    raw = model
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # ---------------- pos_weight ----------------
    def pw(y):
        p = float(y.mean())
        return min((1 - p) / max(p, 1e-8), args.pw_cap) if p > 0 else 1.0
    yd = train_df['DBP_label'].values
    yr = train_df['RBP_label'].values
    yn = ((yd == 0) & (yr == 0)).astype(float)          # non = 非结合
    W = {'d': torch.tensor(pw(yd), device=device),
         'r': torch.tensor(pw(yr), device=device),
         'b': torch.tensor(pw(yd * yr), device=device),
         'n': torch.tensor(pw(yn), device=device)}

    # ---------------- 日志目录 ----------------
    run_dir = None
    if is_main(rank):
        tag = args.tag or args.mode
        run_dir = os.path.join(RUNS_DIR, f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}")
        os.makedirs(run_dir, exist_ok=True)
        cov_tr = struct_coverage(train_df) if use_struct else (0, len(train_df))
        extra = {**data_stats(train_df, 'train'), **data_stats(val_df, 'val'),
                 'pos_weight_dbp': round(float(W['d']), 3),
                 'pos_weight_rbp': round(float(W['r']), 3),
                 'pos_weight_drbp': round(float(W['b']), 3),
                 'struct_cov_train': f'{cov_tr[0]}/{cov_tr[1]}',
                 'world_size': world_size,
                 'n_params_trainable': sum(p.numel() for p in raw.parameters() if p.requires_grad)}
        write_config(run_dir, args, extra)
        write_arch(run_dir, raw)
        print(f"[run] {run_dir}", flush=True)
        print(f"训练 {len(train_df)} / 验证 {len(val_df)}  mode={args.mode}  "
              f"pos_weight d/r/b = {float(W['d']):.2f}/{float(W['r']):.2f}/{float(W['b']):.2f}",
              flush=True)
        if use_struct:
            print(f"结构缓存命中 {cov_tr[0]}/{cov_tr[1]} ({cov_tr[0]/max(cov_tr[1],1):.1%})", flush=True)
            if cov_tr[0] < 0.5 * cov_tr[1]:
                print("!! 结构缓存命中率过低, joint 会退化成纯序列。先跑 build_cache.py", flush=True)
        with open(os.path.join(run_dir, 'metrics.csv'), 'w', newline='') as f:
            csv.writer(f).writerow(METRIC_COLS)

    # ---------------- loader ----------------
    collate = make_collate(tokenizer.pad_token_id, use_struct)

    def loader_for(df, shuffle, distributed=True):
        ds = DRBPDataset(df, tokenizer, args.max_len, use_struct)
        if distributed and world_size > 1:
            sp = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                    shuffle=shuffle, drop_last=shuffle)
            return DataLoader(ds, batch_size=args.batch_size, sampler=sp,
                              collate_fn=collate, num_workers=args.workers)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=collate, num_workers=args.workers)

    train_loader = loader_for(train_df, True)
    val_loader = loader_for(val_df, False)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ---------------- 训练 ----------------
    best_val, best_state, best_ep, bad = -1.0, None, 0, 0   # 选最优用 sel_ema (越大越好)
    best_t = (0.5, 0.5, 0.5)
    sel_prev = None                                          # EMA 状态

    for epoch in range(1, args.epochs + 1):
        model.train()
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        tot, nb, parts = 0.0, 0, {}
        n_batches = len(train_loader)
        opt.zero_grad()
        for i, batch in enumerate(train_loader):
            b = move(batch, device)
            # 梯度累积: joint 模式下 batch>4 会 OOM (最坏情况整批 1024 残基占 6.7GB),
            # 用累积把有效 batch 拉回和 esm2 一致, 否则消融被 batch size 混杂
            is_step = ((i + 1) % args.accum == 0) or (i + 1 == n_batches)
            # 非同步步不做 all-reduce, 省通信
            ctx = (model.no_sync() if (world_size > 1 and not is_step)
                   else contextlib.nullcontext())
            with ctx:
                o = run_model(model, b)
                if args.head_scheme == 'non':
                    loss, pd_ = non_loss(o, b['dbp'], b['rbp'], W['d'], W['r'], W['n'])
                else:
                    loss, pd_ = drbp_loss(o, b['dbp'], b['rbp'], W['d'], W['r'], W['b'],
                                          args.lam1, args.lam2, args.cons_mode,
                                          args.w_cons_pos, args.lam3, args.conj_margin)
                (loss / args.accum).backward()
            if is_step:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
            tot += loss.item()
            nb += 1
            for k, v in pd_.items():
                parts[k] = parts.get(k, 0.0) + v
        train_loss = tot / max(nb, 1)
        parts = {k: v / max(nb, 1) for k, v in parts.items()}
        sched.step()

        val_loss = eval_loss(model, val_loader, device, world_size, W, args)
        dp, rp, bp, dy, ry = collect_predictions(model, val_loader, device, world_size)

        # ---- 选最优模型 (打分口径见 selection_score 的注释)。
        # 搜阈值必须搬到广播之前 —— 选择分里含 macro-F1(搜索阈值下), 它依赖 t_d/t_r/t_b。
        if is_main(rank):
            t_d, t_r, t_b, bmcc = search_thresholds(dp, rp, bp, dy, ry,
                                                    use_drbp_head=(args.head_scheme != 'non'))
            m = compute_metrics(dp, rp, bp, dy, ry, t_d, t_r, t_b,
                                use_drbp_head=(args.head_scheme != 'non'))
            diag = collapse_diagnostics(dp, rp, bp)
            # 固定 0.5 的四类 macro-F1: 保留下来只为和历史 run 对齐, 默认不再用它选模型
            _dy = dy.astype(int); _ry = ry.astype(int)
            _pred = np.zeros(len(_dy), int)
            _true = np.zeros(len(_dy), int)
            _true[(_dy == 1) & (_ry == 0)] = 1
            _true[(_dy == 0) & (_ry == 1)] = 2
            _true[(_dy == 1) & (_ry == 1)] = 3
            _flag = drbp_decision(dp, rp, bp, 0.5, 0.5, 0.5,
                                  use_drbp_head=(args.head_scheme != 'non'))
            _pred[(dp > 0.5) & (rp <= 0.5)] = 1
            _pred[(dp <= 0.5) & (rp > 0.5)] = 2
            _pred[(dp > 0.5) & (rp > 0.5)] = 3
            _pred[_flag] = 3
            val_macro_f1 = macro_f1(_true, _pred)
            sel_raw, mean_auc = selection_score(m, val_macro_f1, args.select_metric)
        else:
            val_macro_f1, sel_raw, mean_auc = 0.0, 0.0, 0.0
        # 广播给所有卡 —— 否则各卡 is_best 不一致, rank0 进测试块等 barrier 而其它卡
        # 直接进下一轮, 触发 NCCL 看门狗超时。
        if world_size > 1:
            t = torch.tensor([sel_raw, val_macro_f1, mean_auc], device=device)
            dist.broadcast(t, src=0)
            sel_raw, val_macro_f1, mean_auc = float(t[0]), float(t[1]), float(t[2])

        # EMA 平滑: 单个 epoch 的运气尖峰不足以抢走 best (joint 的 ep2 就是这么把门槛
        # 顶到天花板的)。所有卡拿同一个广播后的 sel_raw 更新, EMA 状态天然一致。
        # --select_ema 0 关掉。
        sel_ema = (sel_raw if sel_prev is None
                   else args.select_ema * sel_prev + (1 - args.select_ema) * sel_raw)
        sel_prev = sel_ema

        is_best = sel_ema > best_val + 1e-6
        if is_best:
            best_val, best_ep, bad = sel_ema, epoch, 0
            # ESM 冻结时不存它的权重: 148M 参数 / 598MB, 每次 best 都克隆一份太重,
            # 加载时从 from_pretrained 重建即可
            best_state = {k: v.detach().cpu().clone()
                          for k, v in raw.state_dict().items()
                          if not (cfg.freeze_esm and k.startswith('esm.'))}
        else:
            bad += 1

        if is_main(rank):
            if is_best:
                best_t = (t_d, t_r, t_b)
            a_d, a_r = raw.alphas()
            row = {'epoch': epoch, 'train_loss': round(train_loss, 4),
                   'val_loss': round(val_loss, 4), **{k: round(v, 4) for k, v in parts.items()},
                   **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
                   **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in diag.items()},
                   'best_t_d': t_d, 'best_t_r': t_r, 'best_t_b': t_b,
                   'best_mcc': round(bmcc, 4),
                   'val_macro_f1': round(val_macro_f1, 4),
                   'macro_f1_srch': round(m['macro_f1'], 4),
                   'mean_auc': round(mean_auc, 4),
                   'sel_raw': round(sel_raw, 4), 'sel_ema': round(sel_ema, 4),
                   'alpha_d': round(a_d, 4), 'alpha_r': round(a_r, 4),
                   'lr': f"{sched.get_last_lr()[0]:.2e}", 'best': int(is_best)}
            with open(os.path.join(run_dir, 'metrics.csv'), 'a', newline='') as f:
                csv.writer(f).writerow([row.get(c, '') for c in METRIC_COLS])
            print(f"Epoch {epoch:3d}/{args.epochs} train={train_loss:.4f} val={val_loss:.4f} "
                  f"AUC d/r/b={m['dbp_auc']:.3f}/{m['rbp_auc']:.3f}/{m['drbp_auc']:.3f} "
                  f"MCC={m['mcc']:.3f} DRBP={m['drbp_caught']}/{m['drbp_total']}"
                  f"(pred {m['drbp_pred']}, P={m['drbp_prec']:.2f}) "
                  f"coact={m['coact']:.2f}(弱头p={m['med_p_weak']:.2f}) "
                  f"sel={sel_raw:.4f}/ema {sel_ema:.4f} "
                  f"corr_dr={diag['corr_dr']:.3f} a=({a_d:+.3f},{a_r:+.3f}) "
                  f"t=({t_d:.2f},{t_r:.2f},{t_b:.2f}){' ★' if is_best else ''}", flush=True)

            # 每个 epoch 存一份 head 权重 (~6MB, 冻结的 ESM 不在里面, 30 epoch 约 180MB)。
            # 有了它, 以后想换选择口径可以直接事后重选, 不用为选错 epoch 重训一遍。
            if args.save_every_epoch:
                ep_dir = os.path.join(run_dir, 'epochs')
                os.makedirs(ep_dir, exist_ok=True)
                torch.save({'state_dict': {k: v.detach().cpu()
                                           for k, v in raw.state_dict().items()
                                           if not (cfg.freeze_esm and k.startswith('esm.'))},
                            'esm_excluded': cfg.freeze_esm, 'epoch': epoch,
                            'thresholds': (t_d, t_r, t_b), 'metrics': row},
                           os.path.join(ep_dir, f'ep{epoch:02d}.pt'))

        # 每次出新最优就在 4 个测试集上评估一次, 累积到 test_history.csv。
        # 只有 rank0 干活 (用 unwrapped 的 raw, 没有集合通信), 其它 rank 在 barrier 等。
        if args.test_every_best and is_best:
            if is_main(rank):
                rows = evaluate_test_sets(raw, tokenizer, collate, args, device,
                                          use_struct, best_t, run_dir, epoch)
                append_test_rows(run_dir, rows)
                print_test_table([r for r in rows if r['thr'] == 'searched'],
                                 f"epoch {epoch} 最优模型 · 测试集 (搜到的阈值)")
            if world_size > 1:
                dist.barrier()

        if bad >= args.patience:
            if is_main(rank):
                print(f"早停 (最优 epoch {best_ep}, "
                      f"{args.select_metric}[ema]={best_val:.4f})", flush=True)
            break

    # ---------------- 测试 (只在 rank0, 普通 DataLoader) ----------------
    if best_state is not None:
        raw.load_state_dict(best_state, strict=False)   # strict=False: ESM 权重不在里面

    if is_main(rank):
        # 存 best_state 而不是 raw.state_dict() —— 后者含 148M 冻结的 ESM 参数(600MB)
        save_sd = best_state if best_state is not None else {
            k: v for k, v in raw.state_dict().items()
            if not (cfg.freeze_esm and k.startswith('esm.'))}
        torch.save({'state_dict': save_sd, 'args': vars(args),
                    'esm_excluded': cfg.freeze_esm,
                    'best_epoch': best_ep, 'best_val': best_val,
                    'select_metric': args.select_metric, 'select_ema': args.select_ema,
                    'best_thresholds': best_t},
                   os.path.join(run_dir, 'best.pt'))
        # 最终评估 (用最优权重), 单独写一份 test_results.csv
        rows = evaluate_test_sets(raw, tokenizer, collate, args, device,
                                  use_struct, best_t, run_dir, best_ep)
        append_test_rows(run_dir, rows, 'test_results.csv')
        print_test_table(rows, f"最终 · 最优 epoch {best_ep} "
                               f"({args.select_metric}[ema]={best_val:.4f}) "
                               f"· 阈值 t=({best_t[0]:.2f},{best_t[1]:.2f},{best_t[2]:.2f})")
        print(f"\n结果已写入 {run_dir}", flush=True)

    if world_size > 1:
        dist.barrier()          # 其它 rank 等 rank0 跑完测试, 否则提前退出会挂
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['esm2', 'joint', 'struct'], default='esm2')
    p.add_argument('--tag', type=str, default=None)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--accum', type=int, default=2,
                   help='梯度累积步数。有效 batch = batch_size × accum × n_gpus')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--head_hidden', type=int, default=64)
    p.add_argument('--max_len', type=int, default=1024)
    p.add_argument('--esm_model', type=str, default="facebook/esm2_t30_150M_UR50D")
    p.add_argument('--unfreeze_esm', action='store_true')
    p.add_argument('--unfreeze_esm_layers', type=int, default=0,
                   help='解冻 ESM 最后 N 层微调 (0=全冻结, 6=解冻后6层)')
    # 损失
    p.add_argument('--lam1', type=float, default=1.0, help='DRBP 直接监督权重')
    p.add_argument('--lam2', type=float, default=0.3,
                   help='层级一致性权重。这是 DRBP 头把知识传给两个主头的唯一直接通路 '
                        '(最终判定只用主头), 光调大 pw_cap 不调它没用')
    p.add_argument('--pw_cap', type=float, default=40.0,
                   help='pos_weight 上限。DRBP 原始值 56.9 (完全平衡), 40 时正样本约占 '
                        '41%% 梯度, 20 时只占 26%%')
    p.add_argument('--cons_mode', choices=['teacher', 'sym'], default='teacher',
                   help='teacher: detach DRBP 头, 单向教主头; sym: 对称 MSE(会把 DRBP 头拽下来)')
    # 共现修复 (见 model_drbp.drbp_loss 的注释)。两个都设 0 = 完全旧行为, 用作消融对照。
    p.add_argument('--w_cons_pos', type=float, default=1.0,
                   help='cons 项额外加一份"只在真 DRBP 上"的误差, 抵消 98.3%% 负样本对这一'
                        '项的主导。0 = 旧行为(全 batch 平均, 实际在教两个主头互斥)')
    p.add_argument('--lam3', type=float, default=0.5,
                   help='共现边际损失权重: 真 DRBP 上惩罚较弱的那个头。0 = 关闭')
    p.add_argument('--conj_margin', type=float, default=1.0,
                   help='共现边际(logit 空间)。1.0 对应 σ(1.0)≈0.73, 即要求两个主头都到 0.73')
    p.add_argument('--no_crosstalk', action='store_true')
    p.add_argument('--no_cross_label', action='store_true',
                   help='关掉 cross-label attention (标签间注意力)。保留三个独立 MLP 头')
    p.add_argument('--no_gate', action='store_true',
                   help='关掉共激活门控 f_d/f_r (消融: 门控 vs 无门控)')
    p.add_argument('--cla_heads', type=int, default=4,
                   help='cross-label attention 的头数')
    p.add_argument('--head_scheme', choices=['drbp', 'non'], default='drbp',
                   help='drbp: 第三头=DRBP(稀有类, 需要过采样); non: 第三头=non(非结合, '
                        '三头全常见类, DRBP 由 DBP∧RBP 派生 + 共激活门控 f)')
    p.add_argument('--test_every_best', action='store_true', default=True,
                   help='每次出现新最优就在 4 个测试集上评估一次')
    p.add_argument('--no_test_every_best', dest='test_every_best', action='store_false',
                   help='关掉中途评估(每次约 6-8 分钟)。每 epoch 权重都存着, 可事后补评估')
    # 选最优 checkpoint
    p.add_argument('--select_metric', choices=SELECT_METRICS, default='composite',
                   help='选最优 checkpoint 的口径, 见 selection_score(). composite = '
                        '0.5×mean(三头AUC) + 0.5×macro-F1(搜索阈值下)。macro_f1_fix 是旧'
                        '行为(固定0.5), 噪声是信号的两倍, 只为复现历史 run 保留')
    p.add_argument('--select_ema', type=float, default=0.5,
                   help='选择分的 EMA 系数, 0=关掉。0.5 约等于 2-epoch 平滑, 防止单个 '
                        'epoch 的运气尖峰独占 best')
    p.add_argument('--save_every_epoch', action='store_true', default=True,
                   help='每 epoch 存一份 head 权重到 runs/*/epochs/ (~6MB/个)。有了它, '
                        '以后换选择口径可以事后重选, 不用为选错 epoch 重训')
    p.add_argument('--no_save_every_epoch', dest='save_every_epoch', action='store_false')
    # 结构
    p.add_argument('--gnn_dim', type=int, default=256)
    p.add_argument('--gnn_layers', type=int, default=4)
    p.add_argument('--gnn_k', type=int, default=16)
    # 其它
    p.add_argument('--val_size', type=float, default=0.1)
    p.add_argument('--drbp_upsample', type=float, default=1.0,
                   help='DRBP 过采样倍率 (训练集内)。1.0=不采样。5.0 → DRBP 占比约 8%。'
                        '数据层面解决 DRBP 稀有类问题 (1.7% → 概率校准坏)。验证集不受影响')
    p.add_argument('--patience', type=int, default=12)   # EMA 有约 1 epoch 滞后, 放宽
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--train_csv', type=str, default="train_final.csv")
    p.add_argument('--test_sets', type=str, nargs='+',
                   default=['DRBP206', 'PDB255', 'TEST474', 'EZL'])
    p.add_argument('--n_gpus', type=int, default=1)
    args = p.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)
    if args.n_gpus > 1 and 'RANK' not in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(_find_free_port())
        print(f"[Launcher] fork 启动 {args.n_gpus} 卡 (PORT={os.environ['MASTER_PORT']})",
              flush=True)
        mp.start_processes(main_worker, args=(args.n_gpus, args), nprocs=args.n_gpus,
                           start_method='fork')
        sys.exit(0)
    main_worker(None, None, args)


if __name__ == "__main__":
    main()
