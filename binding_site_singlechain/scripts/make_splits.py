#!/usr/bin/env python3
"""去冗余后生成训练/验证划分（单链版）。

- 保留训练集 = train_keep_ids.txt ∩ singlechain_labels_aug80.csv（不做长度过滤）。
- 验证集 = 从保留训练集随机抽 10% (seed 42)，其余 90% 训练。
- 产出 data/labels.csv (train+val) + data/split.csv (pdb_id, split)。

测试集不在这里合并 (test_labels.csv 的 pdb_id 形如 3jcm_B，与单链 BioLiP 命名一致
但来源不同)，由 eval_sliding.py 单独读取。

用法: cd /root/DRBP/finetune_6layers/binding_site_singlechain && \
      /root/.conda/envs/drbp/bin/python scripts/make_splits.py
"""
import os, argparse
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, 'data')
KEEP = os.path.join(DATA, 'train_keep_ids.txt')
BIOLIP = os.path.join(DATA, 'singlechain_labels_aug80.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--biolip', default=BIOLIP,
                    help='单链 BioLiP 标注 CSV (默认 aug80)')
    ap.add_argument('--keep', default=KEEP,
                    help='去冗余保留 id 文件 (默认 train_keep_ids.txt)')
    ap.add_argument('--out_dir', default=DATA,
                    help='输出目录 (labels.csv/split.csv 所在)')
    args = ap.parse_args()
    keep = set(l.strip() for l in open(args.keep) if l.strip())
    print(f'去冗余后保留候选: {len(keep)}', flush=True)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_labels = os.path.join(out_dir, 'labels.csv')
    out_split = os.path.join(out_dir, 'split.csv')

    biolip = pd.read_csv(args.biolip, dtype={'pdb_id': str})
    biolip['pdb_id'] = biolip['pdb_id'].astype(str).str.strip()
    df = biolip[biolip['pdb_id'].isin(keep)].reset_index(drop=True)
    print(f'命中单链 labels: {len(df)}', flush=True)

    # 不做长度过滤 (按用户要求)

    # 90/10 切分 (seed 42)
    train, val = train_test_split(df, test_size=0.1, random_state=42)
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    print(f'train {len(train)} / val {len(val)}', flush=True)

    labels = pd.concat([train, val], ignore_index=True)
    labels.to_csv(out_labels, index=False)

    split = pd.concat([
        pd.DataFrame({'pdb_id': train['pdb_id'], 'split': 'train'}),
        pd.DataFrame({'pdb_id': val['pdb_id'], 'split': 'val'}),
    ], ignore_index=True)
    split.to_csv(out_split, index=False)

    def haspos(s):
        return isinstance(s, str) and bool(s.strip())

    n_dbp_tr = int(train['dna_positions'].apply(haspos).sum())
    n_rbp_tr = int(train['rna_positions'].apply(haspos).sum())
    n_dbp_va = int(val['dna_positions'].apply(haspos).sum())
    n_rbp_va = int(val['rna_positions'].apply(haspos).sum())
    print(f'\n训练集 DBP {n_dbp_tr} / RBP {n_rbp_tr}', flush=True)
    print(f'验证集 DBP {n_dbp_va} / RBP {n_rbp_va}', flush=True)
    print(f'\n产物: {out_labels} 和 {out_split}', flush=True)


if __name__ == '__main__':
    main()
