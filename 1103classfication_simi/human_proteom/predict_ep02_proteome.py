#!/usr/bin/env python3
"""withsimi ep02 人类蛋白组 DBP/RBP 预测 + 严格 GO 多阈值召回。

- 模型 = classfication_simi 的 ep02 (train_withsimi 22181, head_scheme='two', 无 NDRBP 头)。
- 参考集 = 严格 GO (GO:0003677 DBP=2267 / GO:0003723 RBP=1598, reviewed+human)。
- 多阈值召回, 便于和 baseline best_twoquery(0.85/0.75) 与 clean30 ep03 对比。

用法:
  3 卡分片 (cuda:1/2/3):
    for g in 1 2 3; do python predict_ep02_proteome.py --shard $((g-1)) 3 --device cuda:$g & done; wait
    python predict_ep02_proteome.py --merge
"""
import os, sys, argparse, glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
SIMI_DIR = os.path.dirname(HERE)
sys.path.insert(0, SIMI_DIR)
from model_baseline import Config
from model_drbp import DRBPNetNew
from train_drbp import DRBPDataset, make_collate, run_model

PROTEOME = '/root/DRBP/TriPepSVM-master/paper/data/proteome_9606.fasta'
ESM_MODEL = '/root/DRBP/models/esm2_t33_650M_UR50D'
CKPT = os.path.join(SIMI_DIR, 'runs', '20261102_235135_no_non_650M_withsimi',
                    'epochs', 'ep02.pt')
TRAIN_CSV = '/root/DRBP/new/data/parsed/train_withsimi.csv'
REF_DIR = '/root/final_DRBP/classfication_twoquery/human_proteome'
REF_DBP = os.path.join(REF_DIR, 'ref_human_dbp_strictGO.txt')
REF_RBP = os.path.join(REF_DIR, 'ref_human_rbp_strictGO.txt')
PRED_CSV = os.path.join(HERE, 'proteome_predictions_ep02_withsimi.csv')
STATS_TXT = os.path.join(HERE, 'proteome_stats_ep02_strictGO.txt')

# 阈值对 (dbp, rbp): 基线 0.85/0.75 + withsimi ep02 自身 val 搜索 0.45/0.95 + 更松的
THRESHOLDS = [(0.85, 0.75), (0.45, 0.95), (0.65, 0.45), (0.50, 0.50), (0.30, 0.30)]


def parse_proteome(path):
    rows, cur_id, cur_entry, cur_seq = [], None, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith('>'):
            if cur_id is not None:
                rows.append((cur_id, cur_entry, ''.join(cur_seq)))
            parts = line[1:].split('|')
            cur_id = parts[1] if len(parts) > 1 else line[1:].split()[0]
            cur_entry = parts[2].split()[0] if len(parts) > 2 else ''
            cur_seq = []
        else:
            cur_seq.append(line)
    if cur_id is not None:
        rows.append((cur_id, cur_entry, ''.join(cur_seq)))
    df = pd.DataFrame(rows, columns=['protein_id', 'entry', 'seq'])
    df['seq_len'] = df.seq.str.len()
    return df


def load_model(ckpt_path, device):
    from transformers import EsmTokenizer, EsmModel
    tok = EsmTokenizer.from_pretrained(ESM_MODEL, local_files_only=True)
    esm = EsmModel.from_pretrained(ESM_MODEL, local_files_only=True)
    cfg = Config(esm_model_name=ESM_MODEL, esm_embed_dim=esm.config.hidden_size,
                 freeze_esm=False, d_model=256, n_heads=8, dropout=0.3, max_seq_len=1024)
    cfg.mode = 'esm2'; cfg.head_hidden = 64; cfg.head_scheme = 'two'
    cfg.use_crosstalk = True; cfg.use_cross_label = True; cfg.cla_heads = 4
    cfg.use_gate = False
    model = DRBPNetNew(cfg, esm, None)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    missing, unexpected = None, None
    try:
        res = model.load_state_dict(ck['state_dict'], strict=True)
        missing, unexpected = res
    except Exception:
        res = model.load_state_dict(ck['state_dict'], strict=False)
        missing, unexpected = res
        print(f"[warn] strict=False 加载: 缺 {len(missing)} 个 key, 多 {len(unexpected)} 个 key",
              flush=True)
    model.to(device).eval()
    return model, tok


def predict(device, batch_size, df, out_csv, shard_id=None, n_shards=None):
    model, tok = load_model(CKPT, device)
    print(f"加载 {CKPT} (650M, head_scheme='two')", flush=True)

    sub = df
    if shard_id is not None:
        sub = df.iloc[shard_id::n_shards].reset_index(drop=True)
        print(f"[shard {shard_id}/{n_shards}] 处理 {len(sub)} 条", flush=True)

    sub = sub.copy()
    sub['DBP_label'] = 0; sub['RBP_label'] = 0
    ds = DRBPDataset(sub, tok, max_len=1024, use_struct=False)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=make_collate(tok.pad_token_id, use_struct=False))

    dp, rp = [], []
    with torch.no_grad():
        for bi, b in enumerate(ld):
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            o = run_model(model, b)
            dp.append(o['dbp_prob'].float().cpu().numpy())
            rp.append(o['rbp_prob'].float().cpu().numpy())
            if bi % 200 == 0:
                print(f"  batch {bi}/{len(ld)}", flush=True)
    dp = np.concatenate(dp); rp = np.concatenate(rp)

    out = sub[['protein_id', 'entry', 'seq']].copy()
    out.columns = ['id', 'entry', 'sequence']
    out['dbp_prob'] = dp.round(4)
    out['rbp_prob'] = rp.round(4)
    out.to_csv(out_csv, index=False)
    print(f"保存 {out_csv}: {len(out)} 条", flush=True)
    del model
    torch.cuda.empty_cache()


def compute_stats(pred_csv, stats_txt):
    out = pd.read_csv(pred_csv)
    n = len(out)
    ref_dbp = set(l.strip() for l in open(REF_DBP) if l.strip())
    ref_rbp = set(l.strip() for l in open(REF_RBP) if l.strip())
    train_ids = set(pd.read_csv(TRAIN_CSV, usecols=['protein_id']).protein_id.astype(str))

    known_d = out[out.id.isin(ref_dbp)]
    known_r = out[out.id.isin(ref_rbp)]
    L = []
    L.append(f"人类蛋白组 DBP/RBP 预测 — withsimi ep02 (head_scheme='two')")
    L.append(f"严格 GO 参考集: DBP {len(ref_dbp)} / RBP {len(ref_rbp)}")
    L.append(f"蛋白组内命中: DBP {len(known_d)} / RBP {len(known_r)}")
    L.append(f"蛋白组总条数: {n}")
    L.append("")
    L.append(f"{'阈值(d/r)':<16}{'DBP召回':>10}{'RBP召回':>10}{'DBP精率':>10}{'RBP精率':>10}")
    for t_d, t_r in THRESHOLDS:
        pd_ = (out.dbp_prob > t_d).astype(int)
        pr_ = (out.rbp_prob > t_r).astype(int)
        n_d = int(pd_.sum()); n_r = int(pr_.sum())
        d_rec = int((known_d.dbp_prob > t_d).sum())
        r_rec = int((known_r.rbp_prob > t_r).sum())
        d_rec_pct = d_rec / len(known_d) * 100 if len(known_d) else 0
        r_rec_pct = r_rec / len(known_r) * 100 if len(known_r) else 0
        d_prec = int(out[pd_ == 1].id.isin(ref_dbp).sum()) / n_d * 100 if n_d else 0
        r_prec = int(out[pr_ == 1].id.isin(ref_rbp).sum()) / n_r * 100 if n_r else 0
        L.append(f"{t_d:.2f}/{t_r:.2f}     {d_rec_pct:9.1f}% {r_rec_pct:9.1f}% "
                 f"{d_prec:9.1f}% {r_prec:9.1f}%")

    L.append("")
    L.append("== 干净召回 (阈值 0.85/0.75, 排除训练集同 accession) ==")
    for name, known, probcol, thr in [('DBP', known_d, 'dbp_prob', 0.85),
                                      ('RBP', known_r, 'rbp_prob', 0.75)]:
        out_tr = known[~known.id.isin(train_ids)]
        if len(out_tr):
            rec = int((out_tr[probcol] > thr).sum())
            L.append(f"  {name}: 干净 {len(out_tr)} 条, 命中 {rec} ({rec/len(out_tr)*100:.1f}%)")

    txt = '\n'.join(L)
    print('\n' + txt, flush=True)
    with open(stats_txt, 'w') as f:
        f.write(txt + '\n')
    print(f"\n统计保存 {stats_txt}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--shard', nargs=2, type=int, default=None, metavar=('K', 'N'))
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()

    if args.merge:
        shards = sorted(glob.glob(os.path.join(HERE, 'proteome_ep02_shard*.csv')))
        if not shards:
            print("未找到 shard 文件"); sys.exit(1)
        merged = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
        merged.to_csv(PRED_CSV, index=False)
        print(f"合并 {len(shards)} 片 → {PRED_CSV}: {len(merged)} 条", flush=True)
        compute_stats(PRED_CSV, STATS_TXT)
        return

    device = args.device if torch.cuda.is_available() else 'cpu'
    df = parse_proteome(PROTEOME)
    if args.shard is not None:
        k, n = args.shard
        out_csv = os.path.join(HERE, f'proteome_ep02_shard{k}.csv')
        predict(device, args.batch_size, df, out_csv, shard_id=k, n_shards=n)
    else:
        predict(device, args.batch_size, df, PRED_CSV)
        compute_stats(PRED_CSV, STATS_TXT)


if __name__ == '__main__':
    main()
