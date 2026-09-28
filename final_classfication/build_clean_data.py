#!/usr/bin/env python3
"""
对 22,892 训练集 vs 4 个测试集做 mmseqs 30% 相似性聚类，剔除与测试集相似的训练蛋白。

- 训练 + 测试全部序列放一起 mmseqs easy-cluster (--min-seq-id 0.3 -c 0.8 --cov-mode 1)
- 凡与任一测试蛋白同簇的训练蛋白剔除
- 输出 data/train/train.csv 和 data/test/{DRBP206,PDB255,TEST474,EZL}.csv
  (列 = protein_id, seq, DBP_label, RBP_label)

用法: /root/.conda/envs/drbp/bin/python build_clean_data.py
"""
import os
import subprocess
import pandas as pd

PARSED = '/root/DRBP/new/data/parsed'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data')
MMSEQS = '/usr/local/bin/mmseqs'
TEST_SETS = ['DRBP206', 'PDB255', 'TEST474', 'EZL']
MIN_SEQ_ID = 0.3
COV = 0.8


def dist(df, name):
    d = df['DBP_label']; r = df['RBP_label']
    dbp = ((d == 1) & (r == 0)).sum()
    rbp = ((d == 0) & (r == 1)).sum()
    drbp = ((d == 1) & (r == 1)).sum()
    non = ((d == 0) & (r == 0)).sum()
    print(f'  {name:12s} n={len(df):6d}  DBP={dbp:5d}  RBP={rbp:5d}  DRBP={drbp:4d}  non={non:6d}')


def main():
    train = pd.read_csv(os.path.join(PARSED, 'train_final.csv'))
    tests = {n: pd.read_csv(os.path.join(PARSED, f'{n}.csv')) for n in TEST_SETS}
    all_tests = pd.concat(tests.values(), ignore_index=True)
    print(f'训练集 {len(train)}  测试集 { {n: len(d) for n, d in tests.items()} }')

    # 若 data 仍是 symlink，先移除，建立真实目录（自包含交付）
    if os.path.islink(OUT):
        os.remove(OUT)
    os.makedirs(OUT, exist_ok=True)

    tmp = '/tmp/cls_dered'
    os.makedirs(tmp, exist_ok=True)
    all_fa = os.path.join(tmp, 'all.fasta')
    with open(all_fa, 'w') as f:
        for i, seq in enumerate(train['seq']):
            f.write(f'>tr_{i}\n{seq}\n')
        for j, seq in enumerate(all_tests['seq']):
            f.write(f'>ts_{j}\n{seq}\n')

    prefix = os.path.join(tmp, 'clu')
    subprocess.run([MMSEQS, 'easy-cluster', all_fa, prefix, os.path.join(tmp, 'clutmp'),
                    '--min-seq-id', str(MIN_SEQ_ID), '-c', str(COV), '--cov-mode', '1',
                    '--threads', '8'], check=True)

    test_ids = {f'ts_{j}' for j in range(len(all_tests))}
    clu = prefix + '_cluster.tsv'
    test_reps = set()
    for line in open(clu):
        rep, member = line.rstrip('\n').split('\t')
        if member in test_ids:
            test_reps.add(rep)
    drop = set()
    for line in open(clu):
        rep, member = line.rstrip('\n').split('\t')
        if rep in test_reps and member.startswith('tr_'):
            drop.add(int(member[3:]))
    print(f'测试蛋白 {len(test_ids)} 个, 命中簇 {len(test_reps)} 个, 剔除训练蛋白 {len(drop)} 个')

    keep = train.drop(index=sorted(drop)).reset_index(drop=True)

    os.makedirs(os.path.join(OUT, 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUT, 'test'), exist_ok=True)
    keep[['protein_id', 'seq', 'DBP_label', 'RBP_label']].to_csv(
        os.path.join(OUT, 'train', 'train.csv'), index=False)
    for n in TEST_SETS:
        tests[n][['protein_id', 'seq', 'DBP_label', 'RBP_label']].to_csv(
            os.path.join(OUT, 'test', f'{n}.csv'), index=False)

    print('\n=== 最终组成 ===')
    dist(keep, '训练集')
    for n in TEST_SETS:
        dist(tests[n], f'{n}_Test')
    print(f'\n剔除 {len(train) - len(keep)} / {len(train)} = {(len(train) - len(keep)) / len(train):.2%}')
    print(f'产物: {OUT}/train/train.csv + {OUT}/test/*.csv')


if __name__ == '__main__':
    main()
