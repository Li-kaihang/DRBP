#!/usr/bin/env python3
"""用 650M checkpoint 在 4 个标准测试集上评估 (head_scheme=non, no_gate)。

与 eval_cls.py 的区别: esm_model 用 esm2_t33_650M_UR50D (esm_embed_dim=1280),
use_gate=False (训练时 --no_gate)。评估口径与 train_drbp 的最终评估完全一致。

用法: cd /root/DRBP/finetune_6layers/classfication_version2 && \
      /root/.conda/envs/drbp/bin/python eval_650m_test.py \
          --ckpt runs/20260923_104532_finetune6_650M/epochs/ep03.pt
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_baseline import Config
from model_drbp import DRBPNetNew
from train_drbp import DRBPDataset, make_collate, run_model
from metrics import compute_metrics, search_thresholds

DATA_DIR = '/root/DRBP/new/data/parsed'
ESM_MODEL = '/root/DRBP/models/esm2_t33_650M_UR50D'


def load_model(ckpt_path, device):
    from transformers import EsmTokenizer, EsmModel
    tok = EsmTokenizer.from_pretrained(ESM_MODEL, local_files_only=True)
    esm = EsmModel.from_pretrained(ESM_MODEL, local_files_only=True)
    cfg = Config(esm_model_name=ESM_MODEL, esm_embed_dim=esm.config.hidden_size,
                 freeze_esm=False, d_model=256, n_heads=8, dropout=0.3, max_seq_len=1024)
    cfg.mode = 'esm2'
    cfg.head_hidden = 64
    cfg.head_scheme = 'non'
    cfg.use_crosstalk = True
    cfg.use_cross_label = True
    cfg.cla_heads = 4
    cfg.use_gate = False          # 训练用了 --no_gate
    model = DRBPNetNew(cfg, esm, None)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state_dict'], strict=False)
    model.to(device).eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch_size', type=int, default=8)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(args.ckpt, device)

    print(f"=== {os.path.basename(args.ckpt)} (650M, head_scheme=non, no_gate) ===", flush=True)
    collate = make_collate(tok.pad_token_id, use_struct=False)
    for ts in ['DRBP206', 'PDB255', 'TEST474', 'EZL']:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{ts}.csv"))
        ds = DRBPDataset(df, tok, max_len=1024, use_struct=False)
        ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        dp, rp, bp, dy, ry = [], [], [], [], []
        with torch.no_grad():
            for b in ld:
                b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                o = run_model(model, b)
                dp.append(o['dbp_prob'].float().cpu().numpy())
                rp.append(o['rbp_prob'].float().cpu().numpy())
                bp.append(o['drbp_prob'].float().cpu().numpy())
                dy.append(b['dbp'].cpu().numpy())
                ry.append(b['rbp'].cpu().numpy())
        dp = np.concatenate(dp); rp = np.concatenate(rp); bp = np.concatenate(bp)
        dy = np.concatenate(dy).astype(int); ry = np.concatenate(ry).astype(int)
        t_d, t_r, t_b, bmcc = search_thresholds(dp, rp, bp, dy, ry, use_drbp_head=False)
        m = compute_metrics(dp, rp, bp, dy, ry, t_d, t_r, t_b, use_drbp_head=False)
        cap = f"{m['drbp_caught']}/{m['drbp_total']}" if m['drbp_total'] > 0 else "-"
        print(f"  {ts:<9} dbp_auc={m['dbp_auc']:.4f} rbp_auc={m['rbp_auc']:.4f} "
              f"drbp_auc={m['drbp_auc']:.4f} MCC={m['mcc']:.4f} DRBP={cap}", flush=True)


if __name__ == '__main__':
    main()
