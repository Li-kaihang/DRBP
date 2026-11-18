#!/usr/bin/env python3
"""DRBP-Net（无 NDRBP 头, head_scheme='two'）部署评估脚本。

加载最优 checkpoint，在 4 个独立测试集上报告两类指标：

1. 阈值无关指标 —— DBP / RBP / DRBP 的 ROC AUC（不依赖阈值，可跨模型公平对比）
2. 部署阈值指标 —— 用验证集搜索出的**固定**最优阈值 (t_d=0.85, t_r=0.75)，
   一次性套到测试集上，计算 4 类 MCC / macro-F1 / 各类召回 / DRBP 捕获数与精确率。

   注意与训练时"按每个测试集各自搜阈值"(oracle, 会泄测试集信息) 不同：
   这里是诚实的部署协议 —— 阈值只来自验证集，对测试集直接套用。

用法:
  cd /root/final_DRBP/ablation_no_non/classification_noNDRBP
  /root/.conda/envs/drbp/bin/python test_eval.py \
      --ckpt runs/20261004_064500_no_non_650M/epochs/ep04.pt
"""
import os, sys, argparse, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_baseline import Config
from model_drbp import DRBPNetNew
from train_drbp import DRBPDataset, make_collate, run_model
from metrics import compute_metrics

DATA_DIR = '/root/DRBP/new/data/parsed'
ESM_MODEL = '/root/DRBP/models/esm2_t33_650M_UR50D'
TEST_SETS = ['DRBP206', 'PDB255', 'TEST474', 'EZL']

# 验证集搜索出的部署阈值 (ep04, 4 类 MCC 最优, best_mcc=0.8546)
THR_DBP = 0.85
THR_RBP = 0.75


def load_model(ckpt_path, device):
    from transformers import EsmTokenizer, EsmModel
    tok = EsmTokenizer.from_pretrained(ESM_MODEL, local_files_only=True)
    esm = EsmModel.from_pretrained(ESM_MODEL, local_files_only=True)
    cfg = Config(esm_model_name=ESM_MODEL, esm_embed_dim=esm.config.hidden_size,
                 freeze_esm=False, d_model=256, n_heads=8, dropout=0.3, max_seq_len=1024)
    cfg.mode = 'esm2'
    cfg.head_hidden = 64
    cfg.head_scheme = 'two'       # 无 NDRBP 头
    cfg.use_crosstalk = True
    cfg.use_cross_label = True
    cfg.cla_heads = 4
    cfg.use_gate = False          # 训练用了 --no_gate
    model = DRBPNetNew(cfg, esm, None)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state_dict'], strict=False)
    model.to(device).eval()
    return model, tok


def eval_one(ts, model, tok, device, batch_size):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{ts}.csv"))
    ds = DRBPDataset(df, tok, max_len=1024, use_struct=False)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=make_collate(tok.pad_token_id, use_struct=False))
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
    return dp, rp, bp, dy, ry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--thr_dbp', type=float, default=THR_DBP)
    ap.add_argument('--thr_rbp', type=float, default=THR_RBP)
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    model, tok = load_model(args.ckpt, device)
    print(f"=== {os.path.basename(args.ckpt)} (650M, head_scheme='two', no NDRBP head) ===")
    print(f"=== 部署阈值 (验证集搜索): DBP={args.thr_dbp}  RBP={args.thr_rbp} ===\n")

    results = {}
    header = (f"{'test':<9} {'dbp_auc':>8} {'rbp_auc':>8} {'drbp_auc':>8} "
              f"{'MCC':>7} {'macroF1':>7} {'acc':>6} {'DRBP':>10} {'DRBP_prec':>10}")
    print(header, flush=True)
    for ts in TEST_SETS:
        dp, rp, bp, dy, ry = eval_one(ts, model, tok, device, args.batch_size)
        m = compute_metrics(dp, rp, bp, dy, ry, args.thr_dbp, args.thr_rbp, 0.5,
                            use_drbp_head=False)
        cap = f"{m['drbp_caught']}/{m['drbp_total']}" if m['drbp_total'] > 0 else "-"
        prec = f"{m['drbp_prec']:.3f}" if m['drbp_total'] > 0 else "-"
        print(f"{ts:<9} {m['dbp_auc']:>8.4f} {m['rbp_auc']:>8.4f} {m['drbp_auc']:>8.4f} "
              f"{m['mcc']:>7.4f} {m['macro_f1']:>7.4f} {m['acc']:>6.4f} "
              f"{cap:>10} {prec:>10}", flush=True)
        results[ts] = {
            'dbp_auc': m['dbp_auc'], 'rbp_auc': m['rbp_auc'], 'drbp_auc': m['drbp_auc'],
            'mcc': m['mcc'], 'macro_f1': m['macro_f1'], 'acc': m['acc'],
            'rec_non': m['rec_non'], 'rec_dbp': m['rec_dbp'],
            'rec_rbp': m['rec_rbp'], 'rec_drbp': m['rec_drbp'],
            'drbp_caught': m['drbp_caught'], 'drbp_total': m['drbp_total'],
            'drbp_prec': m['drbp_prec'], 'drbp_rec': m['drbp_rec'],
            't_d': args.thr_dbp, 't_r': args.thr_rbp,
        }

    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump({'ckpt': args.ckpt, 't_d': args.thr_dbp, 't_r': args.thr_rbp,
                       'results': results}, f, indent=2)
        print(f"\n结果已保存 {args.out_json}")


if __name__ == '__main__':
    main()
