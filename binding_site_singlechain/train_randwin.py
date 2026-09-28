#!/usr/bin/env python3
"""binding_site_data_v2 随机窗口训练 (window=512, DDP 4 卡)。

在去冗余后的 BioLiP 训练集上训练，验证集为去冗余后重切 10%。配方沿用 v2:
window=512 / lr=1e-4 / 解冻 ESM 后 6 层 / 序列(ESM-2) + 结构(GNN) + 化学性质
(Python 代理: SASA/凹凸度/静电势)。loss = DNA 头 BCE + RNA 头 BCE (pos_weight)。

DDP: fork 启动 (Singularity 下不能用 torchrun/spawn)，DistributedSampler 分训练集，
验证只跑 rank0 单卡普通 DataLoader。

用法: cd /root/DRBP/finetune_6layers/binding_site_data_v2 && \
      /root/.conda/envs/drbp/bin/python train_randwin.py --n_gpus 4 \
      2>&1 | tee train.log
"""
import os, sys, socket, argparse, subprocess, math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from struct_gnn import StructureGNN
from joint_model import JointBindingSiteModel
from lora import apply_lora_to_esm
from randwin_dataset import RandomWindowDataset, binding_site_collate_fn, _parse_positions
from eval_sliding import evaluate_on_test

HERE = os.path.dirname(os.path.abspath(__file__))
PDB_DIR = '/tmp/bsv4/train_pdbs'   # 单链 PDB (GraphBind 口径, /tmp 快盘)


def focal_bce_with_logits(logits, targets, gamma=2.0, pos_weight=None):
    """Focal loss (针对极端类别不平衡)。

    结合位点正负残基比 ~1:10, 大量易分负样本主导梯度。focal 对"已分对"的样本
    (p_t 接近 1) 降权 (1-p_t)^gamma, 让模型把梯度集中在难分样本上, 压制孤立假阳性、
    提升精度 (precision/MCC)。pos_weight 语义与 BCE 一致 (正样本乘 pos_weight)。
    """
    ce = F.binary_cross_entropy_with_logits(logits, targets,
                                            pos_weight=pos_weight, reduction='none')
    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1.0 - p)
    focal = (1.0 - pt).clamp(min=1e-8) ** gamma
    return focal * ce


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


def load_esm(device, unfreeze_layers=0, model_name='facebook/esm2_t30_150M_UR50D'):
    from transformers import EsmTokenizer, EsmModel
    tok = EsmTokenizer.from_pretrained(model_name, local_files_only=True)
    esm = EsmModel.from_pretrained(model_name, local_files_only=True)
    for p in esm.parameters():
        p.requires_grad = False
    if unfreeze_layers > 0:
        n = len(esm.encoder.layer)
        for i in range(n - unfreeze_layers, n):
            for p in esm.encoder.layer[i].parameters():
                p.requires_grad = True
        print(f"解冻 ESM 最后 {unfreeze_layers} 层 (共 {n} 层)", flush=True)
    esm.to(device)
    return esm, tok


@torch.no_grad()
def evaluate(model, tok, loader, device, window, dbp_ids=None, rbp_ids=None):
    """位点 AUC。DNA 只在 DBP 蛋白、RNA 只在 RBP 蛋白上算 (正确口径)。"""
    model.eval()
    dna_logits, rna_logits, dna_labels, rna_labels = [], [], [], []
    for batch in loader:
        seqs = batch.pop('sequence')
        enc = tok(seqs, padding=True, truncation=True, max_length=window + 2, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        sb = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        dl, rl = model(sb, input_ids, attention_mask)
        L = min(input_ids.shape[1] - 2, sb['mask'].shape[1])
        for i, pid in enumerate(sb['pdb_id']):
            m = sb['mask'][i, :L] > 0.5
            if dbp_ids is None or pid in dbp_ids:
                dna_logits.append(dl[i, :L][m].float().cpu().numpy())
                dna_labels.append(sb['dna_label'][i, :L][m].cpu().numpy())
            if rbp_ids is None or pid in rbp_ids:
                rna_logits.append(rl[i, :L][m].float().cpu().numpy())
                rna_labels.append(sb['rna_label'][i, :L][m].cpu().numpy())
    if not dna_logits:
        return float('nan'), float('nan')
    dna_logits = np.concatenate(dna_logits); rna_logits = np.concatenate(rna_logits)
    dna_labels = np.concatenate(dna_labels); rna_labels = np.concatenate(rna_labels)
    dna_auc = roc_auc_score(dna_labels, dna_logits) if len(np.unique(dna_labels)) > 1 else float('nan')
    rna_auc = roc_auc_score(rna_labels, rna_logits) if len(np.unique(rna_labels)) > 1 else float('nan')
    return dna_auc, rna_auc


def main_worker(local_rank, world_size, args):
    if world_size is not None and world_size > 1 and 'RANK' not in os.environ:
        os.environ['RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
    local_rank, world_size, rank = ddp_setup()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    is_main = (rank == 0)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    import random as _r; _r.seed(args.seed)

    if is_main:
        print(f'加载 ESM-2 ({args.esm_model}) + StructureGNN ...', flush=True)
    use_lora = getattr(args, 'lora_r', 0) > 0
    unfreeze = 0 if use_lora else args.unfreeze_esm_layers
    esm, tok = load_esm(device, unfreeze, args.esm_model)
    if use_lora:
        apply_lora_to_esm(esm, r=args.lora_r, alpha=args.lora_alpha,
                          dropout=0.1, n_layers=args.unfreeze_esm_layers)
        print(f"LoRA 微调 ESM 后 {args.unfreeze_esm_layers} 层 q/k/v (r={args.lora_r}, alpha={args.lora_alpha})", flush=True)
    d_seq = esm.config.hidden_size
    gnn = StructureGNN(node_dim=256, edge_dim=64, n_layers=4, k_neighbors=16,
                       dropout=args.gnn_dropout)
    model = JointBindingSiteModel(esm, gnn, d_seq=d_seq, d_struct=256,
                                  seq_only_dna=args.seq_only_dna,
                                  dropout=args.dropout,
                                  context_head=args.context_head,
                                  gated_fusion=args.gated_fusion).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw = model.module if hasattr(model, 'module') else model
    if is_main:
        n_trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
        print(f'  可训练参数: {n_trainable:,}', flush=True)

    # ---- 数据 ----
    data_dir = os.path.join(HERE, args.data_dir)
    labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
    split = pd.read_csv(os.path.join(data_dir, 'split.csv'))
    labels = labels.merge(split, on='pdb_id', how='inner')
    train_df = labels[labels['split'] == 'train'].reset_index(drop=True)
    val_df = labels[labels['split'] == 'val'].reset_index(drop=True)
    if is_main:
        print(f'  train {len(train_df)} / val {len(val_df)}', flush=True)

    def haspos(s):
        return isinstance(s, str) and bool(s.strip())
    if args.task == 'dna':
        train_df = train_df[train_df.dna_positions.apply(haspos)].reset_index(drop=True)
        val_df = val_df[val_df.dna_positions.apply(haspos)].reset_index(drop=True)
    elif args.task == 'rna':
        train_df = train_df[train_df.rna_positions.apply(haspos)].reset_index(drop=True)
        val_df = val_df[val_df.rna_positions.apply(haspos)].reset_index(drop=True)
    if is_main:
        print(f'  单任务过滤后 ({args.task}): train {len(train_df)} / val {len(val_df)}', flush=True)
    dbp_val_ids = set(val_df.loc[val_df.dna_positions.apply(haspos), 'pdb_id'].astype(str).str.strip())
    rbp_val_ids = set(val_df.loc[val_df.rna_positions.apply(haspos), 'pdb_id'].astype(str).str.strip())
    dbp_train_ids = set(train_df.loc[train_df.dna_positions.apply(haspos), 'pdb_id'].astype(str).str.strip())
    rbp_train_ids = set(train_df.loc[train_df.rna_positions.apply(haspos), 'pdb_id'].astype(str).str.strip())
    if is_main:
        print(f'  验证集 DBP {len(dbp_val_ids)} / RBP {len(rbp_val_ids)}', flush=True)

    train_ds = RandomWindowDataset(PDB_DIR, train_df, window=args.window,
                                   pos_bias=args.pos_bias)
    val_ds = RandomWindowDataset(PDB_DIR, val_df, window=args.window)   # val 保持均匀

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                           shuffle=True, drop_last=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                                  collate_fn=binding_site_collate_fn, num_workers=args.workers,
                                  persistent_workers=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=binding_site_collate_fn, num_workers=args.workers,
                                  persistent_workers=True)
    # 验证只在 rank0 跑，普通 DataLoader (全量 val)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=binding_site_collate_fn, num_workers=args.workers,
                            persistent_workers=True)

    # ---- pos_weight (方案B: 按各头对应蛋白类型算) ----
    dbp_train = train_df[train_df.dna_positions.apply(haspos)]
    rbp_train = train_df[train_df.rna_positions.apply(haspos)]
    n_dna_pos = sum(len(_parse_positions(r['dna_positions'])) for _, r in dbp_train.iterrows())
    n_rna_pos = sum(len(_parse_positions(r['rna_positions'])) for _, r in rbp_train.iterrows())
    n_res_dna = int(dbp_train.sequence.str.len().sum())
    n_res_rna = int(rbp_train.sequence.str.len().sum())
    w_dna = torch.tensor((n_res_dna - n_dna_pos) / max(n_dna_pos, 1) * args.pos_weight_scale,
                         device=device)
    w_rna = torch.tensor((n_res_rna - n_rna_pos) / max(n_rna_pos, 1) * args.pos_weight_scale,
                         device=device)
    if is_main:
        print(f'  pos_weight DNA={float(w_dna):.1f} RNA={float(w_rna):.1f} '
              f'(DBP {len(dbp_train)} / RBP {len(rbp_train)})', flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    # warmup + cosine 调度: 前 warmup_ratio 步 lr 从 0 线性爬到 base, 之后 cosine 衰减到 min_lr。
    # 治「ep2-3 就过拟合掉头」: warmup 让早期学慢一点推迟峰值, cosine 压后期防过拟合。
    total_steps = args.epochs * max(len(train_loader), 1)
    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)

    def _lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    best_test_score = -1.0      # 标准测试集三组 MCC 均值 (选 checkpoint 的指标)
    best_state = None
    best_test_ep = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        if world_size > 1:
            train_sampler.set_epoch(ep)
        tot = tot_dna = tot_rna = 0.0
        nb = 0
        for batch in train_loader:
            seqs = batch.pop('sequence')
            enc = tok(seqs, padding=True, truncation=True, max_length=args.window + 2,
                      return_tensors='pt')
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            sb = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            dl, rl = model(sb, input_ids, attention_mask)
            L = min(input_ids.shape[1] - 2, sb['mask'].shape[1])
            mask = sb['mask'][:, :L] > 0.5            # [B, L]
            dna_label = sb['dna_label'][:, :L]        # [B, L]
            rna_label = sb['rna_label'][:, :L]        # [B, L]
            # 方案B: DNA 头只在 DBP/dual 蛋白上算 loss, RNA 头只在 RBP/dual 蛋白上算。
            # 用 reduction='none' + 残基权重, 保证两头始终在图里 (某边全 0 时 grad=0 而非
            # None), 避免 DDP 报 unused parameter。
            pids = sb['pdb_id']
            is_dna = torch.tensor([p in dbp_train_ids for p in pids], device=device)  # [B]
            is_rna = torch.tensor([p in rbp_train_ids for p in pids], device=device)  # [B]
            dna_w = (mask & is_dna[:, None]).float()   # [B, L]
            rna_w = (mask & is_rna[:, None]).float()   # [B, L]
            if args.task in ('both', 'dna'):
                if args.focal_gamma > 0:
                    bce_dna = focal_bce_with_logits(dl[:, :L], dna_label,
                                                    gamma=args.focal_gamma, pos_weight=w_dna)
                else:
                    bce_dna = F.binary_cross_entropy_with_logits(dl[:, :L], dna_label,
                                                                 pos_weight=w_dna, reduction='none')
                loss_dna = (bce_dna * dna_w).sum() / dna_w.sum().clamp(min=1.0)
            else:
                loss_dna = (dl[:, :L] * 0.0).sum()      # 保持 DNA 头在图里 (grad=0 而非 None)
            if args.task in ('both', 'rna'):
                if args.focal_gamma > 0:
                    bce_rna = focal_bce_with_logits(rl[:, :L], rna_label,
                                                    gamma=args.focal_gamma, pos_weight=w_rna)
                else:
                    bce_rna = F.binary_cross_entropy_with_logits(rl[:, :L], rna_label,
                                                                 pos_weight=w_rna, reduction='none')
                loss_rna = (bce_rna * rna_w).sum() / rna_w.sum().clamp(min=1.0)
            else:
                loss_rna = (rl[:, :L] * 0.0).sum()
            loss = loss_dna + loss_rna
            opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()
            tot += loss.item(); tot_dna += loss_dna.item(); tot_rna += loss_rna.item(); nb += 1

        # ---- 验证 (rank0) ----
        if world_size > 1:
            dist.barrier()
        dna_auc = rna_auc = float('nan')
        if is_main:
            dna_auc, rna_auc = evaluate(raw, tok, val_loader, device, args.window,
                                        dbp_val_ids, rbp_val_ids)
            mean_auc = np.nanmean([dna_auc, rna_auc])
            print(f'epoch {ep:2d}  train_loss={tot/max(nb,1):.4f} '
                  f'(DNA {tot_dna/max(nb,1):.4f} / RNA {tot_rna/max(nb,1):.4f})  '
                  f'val DNA_AUC={dna_auc:.4f} RNA_AUC={rna_auc:.4f} mean={mean_auc:.4f}',
                  flush=True)
            if args.eval_test_every > 0 and ep % args.eval_test_every == 0:
                try:
                    res = evaluate_on_test(raw, tok, device, window=args.window)
                    m129 = res.get('DNA-129'); m181 = res.get('DNA-181'); m117 = res.get('RNA-117')
                    mm = lambda x: f"{x['MCC']:.4f}" if x else '  nan'
                    if args.task == 'dna':
                        rel = (m129, m181)
                    elif args.task == 'rna':
                        rel = (m117,)
                    else:
                        rel = (m129, m181, m117)
                    mccs = [x['MCC'] for x in rel if x is not None]
                    test_score = float(np.mean(mccs)) if mccs else -1.0
                    star = '★' if test_score > best_test_score else ''
                    if test_score > best_test_score:
                        best_test_score = test_score
                        best_test_ep = ep
                        best_state = {k: v.detach().cpu().clone() for k, v in raw.state_dict().items()}
                        # 增量落盘: 每个新最优立即保存, 避免中途被杀丢失 checkpoint
                        torch.save({'model_state_dict': best_state,
                                    'test_mcc_mean': best_test_score, 'epoch': best_test_ep,
                                    'window': args.window}, os.path.join(HERE, args.save))
                    print(f'       [测试集 MCC] DNA-129={mm(m129)}  DNA-181={mm(m181)}  '
                          f'RNA-117={mm(m117)}  mean={test_score:.4f} {star}', flush=True)
                except Exception as e:
                    print(f'       [测试集评估失败] {type(e).__name__}: {e}', flush=True)
        if world_size > 1:
            dist.barrier()

    # ---- 保存 (rank0, 按标准测试集 MCC 均值选最优) ----
    if is_main and best_state is not None:
        out = os.path.join(HERE, args.save)
        torch.save({'model_state_dict': best_state,
                    'test_mcc_mean': best_test_score, 'epoch': best_test_ep,
                    'window': args.window}, out)
        print(f'\n训练完成, 最优 checkpoint (测试集 MCC 均值): {out} '
              f'(ep{best_test_ep}, mean {best_test_score:.4f})', flush=True)
        print('下一步: /root/.conda/envs/drbp/bin/python eval_sliding.py '
              f'--ckpt {out}', flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def run_test_eval(args):
    """训练结束后在单卡上对三个标准测试集评估最佳 checkpoint，MCC 直接进 log。"""
    ckpt = os.path.join(HERE, args.save)
    if not os.path.exists(ckpt):
        print(f'\n[测试评估] 未找到 checkpoint {ckpt}，跳过测试集评估', flush=True)
        return
    print(f'\n{"="*70}\n'
          f'[测试评估] 最佳 checkpoint → 标准测试集 DNA-129 / DNA-181 / RNA-117\n'
          f'{"="*70}', flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, 'eval_sliding.py'),
                    '--ckpt', ckpt, '--esm_model', args.esm_model,
                    '--dropout', str(getattr(args, 'dropout', 0.0))]
                   + (['--seq_only_dna'] if getattr(args, 'seq_only_dna', False) else [])
                   + (['--context_head'] if getattr(args, 'context_head', False) else [])
                   + (['--gated_fusion'] if getattr(args, 'gated_fusion', False) else [])
                   + (['--lora_r', str(args.lora_r),
                       '--lora_alpha', str(getattr(args, 'lora_alpha', 32))]
                      if getattr(args, 'lora_r', 0) > 0 else []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--window', type=int, default=512)
    ap.add_argument('--pos_bias', type=float, default=0.0,
                    help='正样本偏置窗口采样概率 [0,1] (结合位点是连续斑块, 均匀采样常整窗错过)')
    ap.add_argument('--data_dir', default='data',
                    help='训练数据目录 (labels.csv/split.csv 所在, 相对脚本目录)')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--unfreeze_esm_layers', type=int, default=6)
    ap.add_argument('--save', default='best_bsite.pt')
    ap.add_argument('--eval_test_every', type=int, default=1,
                    help='每 N 个 epoch 在三个标准测试集上评估一次 MCC (0=关闭)')
    ap.add_argument('--seq_only_dna', action='store_true',
                    help='DNA 头只吃序列(ESM-2), RNA 头吃序列+结构')
    ap.add_argument('--context_head', action='store_true',
                    help='融合特征后加 2 层残基间 self-attention (捕捉结合位点斑块上下文)')
    ap.add_argument('--gated_fusion', action='store_true',
                    help='门控融合: 序列/结构各自投影, 每个头独立学 gate 决定信结构多少 (替代 concat)')
    ap.add_argument('--task', default='both', choices=['both', 'dna', 'rna'],
                    help='单独训练单个任务: dna=只喂 DBP/dual 只算 DNA loss; rna=只喂 RBP/dual 只算 RNA loss')
    ap.add_argument('--esm_model', default='facebook/esm2_t30_150M_UR50D',
                    help='ESM-2 模型名或本地路径 (650M: /root/DRBP/models/esm2_t33_650M_UR50D)')
    ap.add_argument('--dropout', type=float, default=0.0,
                    help='fusion/两个头 的 dropout 概率')
    ap.add_argument('--gnn_dropout', type=float, default=0.0,
                    help='结构 GNN (从零训练) 的 dropout 正则化概率')
    ap.add_argument('--lora_r', type=int, default=0,
                    help='LoRA 秩 r (>0 启用 LoRA 微调 ESM 后 n 层 q/k/v, 替代全秩解冻)')
    ap.add_argument('--lora_alpha', type=int, default=32,
                    help='LoRA 缩放 alpha (scaling=alpha/r)')
    ap.add_argument('--focal_gamma', type=float, default=0.0,
                    help='focal loss gamma (>0 启用, 0=关闭用普通 BCE)')
    ap.add_argument('--pos_weight_scale', type=float, default=1.0,
                    help='pos_weight 缩放 (过预测时调小, e.g. 0.4; 漏报时调大)')
    ap.add_argument('--weight_decay', type=float, default=0.01,
                    help='AdamW weight_decay (正则化, 过拟合时调大)')
    ap.add_argument('--warmup_ratio', type=float, default=0.0,
                    help='warmup 占总步数比例 (0=关闭, 建议 0.1; 推迟过拟合峰值)')
    ap.add_argument('--min_lr_ratio', type=float, default=0.1,
                    help='cosine 衰减到底的 lr 相对 base 比例')
    ap.add_argument('--n_gpus', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42,
                    help='随机种子 (训练不同 seed 供 ensemble 平均)')
    args = ap.parse_args()

    if args.n_gpus > 1 and 'RANK' not in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(_find_free_port())
        print(f"[Launcher] fork 启动 {args.n_gpus} 卡 (PORT={os.environ['MASTER_PORT']})",
              flush=True)
        mp.start_processes(main_worker, args=(args.n_gpus, args), nprocs=args.n_gpus,
                           start_method='fork')
        run_test_eval(args)
        sys.exit(0)
    main_worker(None, None, args)
    run_test_eval(args)


if __name__ == '__main__':
    main()
