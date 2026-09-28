#!/usr/bin/env python3
"""GraphBind 标注迁移 —— 第 1 步：序列聚类 (seq identity > 0.8)。

做法（比 foldseek 全对全快得多）：
  1. mmseqs easy-cluster 对 44724 条单链按 seq>0.8 (双向覆盖 0.8) 聚类；
  2. 每簇取**最长链**为 representative（GraphBind 口径）；
  3. 簇内成员标注通过序列对齐(带 gap)映射到 rep，做**并集补齐**；
  4. 只为 rep 提取单链 PDB（不再为全部 44724 条提取，省大量 IO）。

产物 (data/01.seq_cluster/)：
  - clusters.tsv      簇号 \t rep_id \t 成员(逗号分隔)
  - aug80.csv         rep_id, sequence, dna_positions, rna_positions (并集)
  - summary.txt       各阶段数目

用法: cd /root/DRBP/finetune_6layers/binding_site_singlechain && \
      /root/.conda/envs/drbp/bin/python scripts/01_seq_cluster.py
"""
import os
import sys
import argparse
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_common import (DATA, LABELS, ORIG_PDB, MMSEQS, TMP, REP_PDB_DIR,
                             SEQ_ID_AT, run, parse_pos, fmt_pos,
                             global_alignment_map, extract_single_chain_by_seq,
                             write_fasta, count_dbp_rbp)

OUT_DIR = os.path.join(DATA, '01.seq_cluster')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default=LABELS)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REP_PDB_DIR, exist_ok=True)
    tmp = os.path.join(TMP, '01_seq')
    os.makedirs(tmp, exist_ok=True)

    df = pd.read_csv(args.labels, dtype={'pdb_id': str})
    df['pdb_id'] = df['pdb_id'].astype(str).str.strip()
    n_in = len(df)
    print(f'输入单链数: {n_in}', flush=True)

    # ---- 1. mmseqs 序列聚类 (seq>0.8, 双向覆盖 0.8) ----
    fasta = os.path.join(tmp, 'all.fasta')
    write_fasta(fasta, list(zip(df['pdb_id'], df['sequence'])))
    pre = os.path.join(tmp, 'seq80')
    run([MMSEQS, 'easy-cluster', fasta, pre, os.path.join(tmp, 't1'),
         '--min-seq-id', str(SEQ_ID_AT), '-c', '0.8', '--cov-mode', '0',
         '--threads', str(args.threads)])

    # 读簇 (rep \t member)，重组为 {mmseqs_rep: [members]}
    mm_clusters = defaultdict(list)
    for line in open(pre + '_cluster.tsv'):
        rep, member = line.rstrip('\n').split('\t')
        mm_clusters[rep].append(member)
    print(f'mmseqs 原始簇数: {len(mm_clusters)}', flush=True)

    seq = dict(zip(df['pdb_id'], df['sequence']))
    dna = {p: parse_pos(s) for p, s in zip(df['pdb_id'], df['dna_positions'])}
    rna = {p: parse_pos(s) for p, s in zip(df['pdb_id'], df['rna_positions'])}

    # ---- 2. 每簇取最长 rep + 标注并集迁移 ----
    rows = []
    clusters = []           # (rep_id, [members])
    n_transfer = 0
    for members in mm_clusters.values():
        rep = max(members, key=lambda p: len(seq[p]))     # 最长链
        rep_dna = set(dna[rep])
        rep_rna = set(rna[rep])
        for mem in members:
            if mem == rep:
                continue
            a2b = global_alignment_map(seq[mem], seq[rep])   # member -> rep
            for pos in dna[mem]:
                t = a2b.get(pos)
                if t is not None:
                    rep_dna.add(t)
            for pos in rna[mem]:
                t = a2b.get(pos)
                if t is not None:
                    rep_rna.add(t)
            n_transfer += 1
        rows.append({'pdb_id': rep, 'sequence': seq[rep],
                     'dna_positions': fmt_pos(rep_dna),
                     'rna_positions': fmt_pos(rep_rna)})
        clusters.append((rep, members))

    out = pd.DataFrame(rows).sort_values('pdb_id').reset_index(drop=True)

    # ---- 3. 只为 rep 提取单链 PDB ----
    n_pdb = 0
    n_miss = 0
    for rep in out['pdb_id']:
        pdb_id = rep.rsplit('_', 1)[0]
        src = os.path.join(ORIG_PDB, f'{pdb_id}.pdb')
        dst = os.path.join(REP_PDB_DIR, f'{rep}.pdb')
        if not os.path.exists(src):
            n_miss += 1
            continue
        if extract_single_chain_by_seq(src, seq[rep], dst):
            n_pdb += 1
        else:
            n_miss += 1

    # ---- 写产物 ----
    out.to_csv(os.path.join(OUT_DIR, 'aug80.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'clusters.tsv'), 'w') as f:
        for i, (rep, members) in enumerate(sorted(clusters)):
            f.write(f'{i}\t{rep}\t{",".join(members)}\n')

    n_dbp, n_rbp = count_dbp_rbp(out)
    summary = (
        f'输入单链数: {n_in}\n'
        f'mmseqs seq>{SEQ_ID_AT} 簇数: {len(mm_clusters)}\n'
        f'representative (最长链): {len(out)}\n'
        f'标注迁移成员数: {n_transfer}\n'
        f'DBP(有 dna 位点): {n_dbp}\n'
        f'RBP(有 rna 位点): {n_rbp}\n'
        f'rep 单链 PDB 提取成功: {n_pdb}  (缺结构跳过: {n_miss})\n'
    )
    print('\n=== 01 汇总 ===', flush=True)
    print(summary, flush=True)
    with open(os.path.join(OUT_DIR, 'summary.txt'), 'w') as f:
        f.write(summary)
    print(f'产物: {OUT_DIR}/aug80.csv, clusters.tsv, summary.txt', flush=True)
    print(f'rep 单链 PDB: {REP_PDB_DIR}/  ({n_pdb} 个)', flush=True)


if __name__ == '__main__':
    main()
