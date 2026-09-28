#!/usr/bin/env python3
"""去冗余 —— 变体 B：序列 30% + 结构 TM-score>0.5。

在 03 的基础上，额外剔除「与任一测试链 TM-score>0.5」的训练链 (foldseek)。

产物 (data/04.seq30_tm05/)：
  - labels.csv    pdb_id, sequence, dna_positions, rna_positions
  - keep_ids.txt  保留的训练链 pdb_id
  - summary.txt   数目

用法: cd /root/DRBP/finetune_6layers/binding_site_singlechain && \
      /root/.conda/envs/drbp/bin/python scripts/04_seq30_tm05.py
"""
import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_common import (DATA, TEST_LABELS, TEST_PDB, FOLDSEEK, TMP,
                             REP_PDB_DIR, TM_DR, cdhit_and_seq_dedup,
                             run, count_dbp_rbp)

OUT_DIR = os.path.join(DATA, '04.seq30_tm05')
FINAL = os.path.join(DATA, '01.seq_cluster', 'aug80.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', default=FINAL)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(TMP, '04_tm05')
    os.makedirs(tmp, exist_ok=True)

    train = pd.read_csv(args.final, dtype={'pdb_id': str})
    train['pdb_id'] = train['pdb_id'].astype(str).str.strip()
    test = pd.read_csv(TEST_LABELS, dtype={'pdb_id': str})
    test['pdb_id'] = test['pdb_id'].astype(str).str.strip()
    print(f'训练代表链 {len(train)} | 测试链 {len(test)}', flush=True)

    # ---- 1+2. 序列 30% 去冗余 (同 03) ----
    kept, stats = cdhit_and_seq_dedup(train, test, tmp, args.threads)

    # ---- 3. test-train 结构 TM>0.5 去冗余 (foldseek) ----
    rep_pdbs = os.path.join(tmp, 'rep_pdbs')
    os.makedirs(rep_pdbs, exist_ok=True)
    n_missing = 0
    for pid in kept['pdb_id']:
        src = os.path.join(REP_PDB_DIR, pid + '.pdb')
        dst = os.path.join(rep_pdbs, pid + '.pdb')
        if os.path.exists(src):
            if os.path.lexists(dst):
                os.remove(dst)          # 清掉上一轮残留的符号链接
            os.symlink(src, dst)
        else:
            n_missing += 1

    testDB = os.path.join(tmp, 'testDB')
    repDB = os.path.join(tmp, 'repDB')
    run([FOLDSEEK, 'createdb', TEST_PDB, testDB])
    run([FOLDSEEK, 'createdb', rep_pdbs, repDB])
    run([FOLDSEEK, 'search', testDB, repDB, os.path.join(tmp, 'structAln'),
         os.path.join(tmp, 'structtmp'), '--max-seqs', '10000',
         '--threads', str(args.threads), '-a'])
    res_m8 = os.path.join(tmp, 'struct.m8')
    run([FOLDSEEK, 'convertalis', testDB, repDB, os.path.join(tmp, 'structAln'),
         res_m8, '--format-output', 'query,target,qtmscore,ttmscore', '--threads', str(args.threads)])

    drop_tm = set()
    for line in open(res_m8):
        parts = line.rstrip('\n').split('\t')
        if len(parts) >= 4 and float(parts[2]) > TM_DR and float(parts[3]) > TM_DR:
            drop_tm.add(parts[1])

    keep_ids = [p for p in kept['pdb_id'] if p not in drop_tm]
    final_keep = kept[kept['pdb_id'].isin(keep_ids)].reset_index(drop=True)

    final_keep.to_csv(os.path.join(OUT_DIR, 'labels.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'keep_ids.txt'), 'w') as f:
        for p in final_keep['pdb_id']:
            f.write(p + '\n')

    n_dbp, n_rbp = count_dbp_rbp(final_keep)
    summary = (
        f'训练代表链输入: {stats["n_train"]}\n'
        f'train-train CD-HIT 30% 后: {stats["n_cdhit"]}\n'
        f'test-train seq30% 剔除: {stats["n_drop_seq"]}\n'
        f'test-train TM>0.5 剔除: {len(drop_tm)}  (训练链缺结构跳过 {n_missing})\n'
        f'最终训练集: {len(final_keep)}\n'
        f'  DBP: {n_dbp}\n'
        f'  RBP: {n_rbp}\n'
    )
    print('\n=== 04 (序列30%+结构TM0.5) 汇总 ===', flush=True)
    print(summary, flush=True)
    with open(os.path.join(OUT_DIR, 'summary.txt'), 'w') as f:
        f.write(summary)
    print(f'产物: {OUT_DIR}/labels.csv, keep_ids.txt, summary.txt', flush=True)


if __name__ == '__main__':
    main()
