#!/usr/bin/env python3
"""去冗余 —— 变体 A：仅序列 30% (train-train CD-HIT 30% + test-train seq 30%)。

不做结构去冗余。最终训练集 = 01 的代表链 (aug80.csv) 经过纯序列 30% 去冗余后的集合。

产物 (data/03.seq30_only/)：
  - labels.csv    pdb_id, sequence, dna_positions, rna_positions
  - keep_ids.txt  保留的训练链 pdb_id (一行一个)
  - summary.txt   数目

用法: cd /root/DRBP/finetune_6layers/binding_site_singlechain && \
      /root/.conda/envs/drbp/bin/python scripts/03_seq30_only.py
"""
import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_common import (DATA, TEST_LABELS, TMP, cdhit_and_seq_dedup,
                             count_dbp_rbp)

OUT_DIR = os.path.join(DATA, '03.seq30_only')
FINAL = os.path.join(DATA, '01.seq_cluster', 'aug80.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', default=FINAL)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(TMP, '03_seq30')
    os.makedirs(tmp, exist_ok=True)

    train = pd.read_csv(args.final, dtype={'pdb_id': str})
    train['pdb_id'] = train['pdb_id'].astype(str).str.strip()
    test = pd.read_csv(TEST_LABELS, dtype={'pdb_id': str})
    test['pdb_id'] = test['pdb_id'].astype(str).str.strip()
    print(f'训练代表链 {len(train)} | 测试链 {len(test)}', flush=True)

    kept, stats = cdhit_and_seq_dedup(train, test, tmp, args.threads)
    kept.to_csv(os.path.join(OUT_DIR, 'labels.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'keep_ids.txt'), 'w') as f:
        for p in kept['pdb_id']:
            f.write(p + '\n')

    n_dbp, n_rbp = count_dbp_rbp(kept)
    summary = (
        f'训练代表链输入: {stats["n_train"]}\n'
        f'train-train CD-HIT 30% 后: {stats["n_cdhit"]}\n'
        f'test-train seq30% 剔除: {stats["n_drop_seq"]}\n'
        f'最终训练集: {stats["n_keep"]}\n'
        f'  DBP: {n_dbp}\n'
        f'  RBP: {n_rbp}\n'
    )
    print('\n=== 03 (仅序列30%) 汇总 ===', flush=True)
    print(summary, flush=True)
    with open(os.path.join(OUT_DIR, 'summary.txt'), 'w') as f:
        f.write(summary)
    print(f'产物: {OUT_DIR}/labels.csv, keep_ids.txt, summary.txt', flush=True)


if __name__ == '__main__':
    main()
