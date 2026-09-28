#!/usr/bin/env python3
"""GraphBind 单链数据管线 —— 公共常量与工具函数。

被 01~04 四个脚本复用，保持路径/口径一致。
"""
import os
import subprocess
import difflib
from collections import defaultdict

# ---- 路径 ----
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, 'data')

LABELS = os.path.join(DATA, 'singlechain_labels.csv')            # 44724 条单链标签
ORIG_PDB = '/root/DRBP/data/binding_site/pdb'                    # 原始完整 PDB (11723 个)
TEST_LABELS = '/root/DRBP/finetune_6layers/binding_site_data_v2/data/test_labels.csv'
TEST_PDB = '/root/DRBP/finetune_6layers/binding_site_data_v2/data/test_pdbs'   # 424 个单链测试 PDB

MMSEQS = '/usr/local/bin/mmseqs'
FOLDSEEK = '/usr/local/bin/foldseek'

TMP = '/tmp/bsv4'                       # 中间产物全放 /tmp 快盘 (/root 是慢 FUSE)
REP_PDB_DIR = os.path.join(TMP, 'rep_pdbs')   # 聚类代表链的单链 PDB

SEQ_ID_AT = 0.8      # 标注迁移: 序列一致度阈值 (GraphBind)
TM_AT = 0.5          # 标注迁移: TM-score 阈值 (GraphBind)
SEQ_ID_DR = 0.3      # 去冗余: 序列一致度阈值
TM_DR = 0.5          # 去冗余: 结构 TM-score 阈值

# 导入同目录 (binding_site_singlechain/) 的 PDB 解析模块
import sys
sys.path.insert(0, HERE)
from data_processing import parse_pdb_backbone  # noqa: E402


def run(cmd, **kw):
    """打印并执行命令，失败即抛错。"""
    print('+', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        raise RuntimeError('命令失败: ' + ' '.join(cmd))
    return r


def parse_pos(s):
    """'3,7,11' -> {3,7,11}; 空/NaN -> set()。"""
    if s is None:
        return set()
    s = str(s).strip()
    if s in ('', 'nan', 'None'):
        return set()
    return {int(x) for x in s.split(',') if x.strip() != ''}


def fmt_pos(s):
    return ','.join(str(x) for x in sorted(s)) if s else ''


def global_alignment_map(seq_a, seq_b):
    """全局对齐 seq_a -> seq_b，返回 {a_idx: b_idx}（缺口处无映射）。

    用 difflib 找匹配块（autojunk=False），对 >80% 一致的两条链足够可靠，
    与 foldseek 3Di 对齐(qaln/taln)的口径一致：匹配残基下标一一对应。
    """
    sm = difflib.SequenceMatcher(None, seq_a, seq_b, autojunk=False)
    a2b = {}
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            a2b[a + k] = b + k
    return a2b


def extract_single_chain_by_seq(pdb_path, target_seq, out_path):
    """从完整 PDB 里抽出「骨架序列 == target_seq」的那条链，写成单链 PDB。

    匹配用序列精确相等（不依赖 BioLiP 的 chain 命名，最稳）。返回 True/False。
    """
    try:
        parsed = parse_pdb_backbone(pdb_path)
    except Exception:
        return False
    seq = parsed['sequence']
    rids = parsed['residue_ids']          # [(chain_char, resseq), ...]

    chain_idx = defaultdict(list)
    for i, (ch, _) in enumerate(rids):
        chain_idx[ch].append(i)

    target_chain = None
    for ch, idxs in chain_idx.items():
        if ''.join(seq[i] for i in idxs) == target_seq:
            target_chain = ch
            break
    if target_chain is None:
        return False

    with open(pdb_path) as f:
        lines = f.readlines()
    n = 0
    with open(out_path, 'w') as o:
        for line in lines:
            if line.startswith('ENDMDL'):
                break
            if not line.startswith('ATOM '):
                continue
            if line[16] not in (' ', 'A'):
                continue
            if line[21] == target_chain:
                o.write(line)
                n += 1
    return n > 0


def write_fasta(path, pid_seq_pairs):
    """写 fasta，pid 里不能有空格。"""
    with open(path, 'w') as f:
        for pid, s in pid_seq_pairs:
            f.write(f'>{pid}\n{s}\n')


def count_dbp_rbp(df):
    """df 需有 dna_positions / rna_positions 列；返回 (n_dbp, n_rbp)。"""
    n_dbp = int(df['dna_positions'].apply(lambda s: bool(parse_pos(s))).sum())
    n_rbp = int(df['rna_positions'].apply(lambda s: bool(parse_pos(s))).sum())
    return n_dbp, n_rbp


def cdhit_and_seq_dedup(train_df, test_df, tmp_dir, threads=8):
    """GraphBind 序列去冗余 (前两步)，返回 (kept_df, stats_dict)。

      1. train-train CD-HIT 30% (mmseqs easy-cluster 0.3)  → 每簇留 1 代表；
      2. test-train seq 30% → 与测试链同簇(seq>0.3)的训练链剔除。

    train_df/test_df 需有 pdb_id, sequence 列。
    """
    import pandas as pd
    n_train = len(train_df)

    def _easy_cluster(dfs, prefix, tmpname):
        fa = os.path.join(tmp_dir, f'{prefix}.fasta')
        write_fasta(fa, [(r.pdb_id, r.sequence) for r in dfs.itertuples()])
        pre = os.path.join(tmp_dir, prefix)
        run([MMSEQS, 'easy-cluster', fa, pre, os.path.join(tmp_dir, tmpname),
             '--min-seq-id', str(SEQ_ID_DR), '-c', '0.8', '--cov-mode', '1',
             '--threads', str(threads)])
        return pre + '_cluster.tsv'

    # ---- 1. train-train CD-HIT 30% ----
    train_tsv = _easy_cluster(train_df, 'train30', 't1')
    reps = set()
    for line in open(train_tsv):
        rep, _ = line.rstrip('\n').split('\t')
        reps.add(rep)
    train_df = train_df[train_df['pdb_id'].isin(reps)].reset_index(drop=True)
    n_cdhit = len(train_df)

    # ---- 2. test-train seq 30% ----
    all_df = pd.concat([train_df, test_df[['pdb_id', 'sequence']]], ignore_index=True)
    all_tsv = _easy_cluster(all_df, 'seqclu', 't2')
    test_ids = set(test_df['pdb_id'].tolist())
    train_ids = set(train_df['pdb_id'].tolist())
    test_reps = set()
    for line in open(all_tsv):
        rep, member = line.rstrip('\n').split('\t')
        if member in test_ids:
            test_reps.add(rep)
    drop_seq = set()
    for line in open(all_tsv):
        rep, member = line.rstrip('\n').split('\t')
        if rep in test_reps and member in train_ids:
            drop_seq.add(member)
    keep = [p for p in train_df['pdb_id'] if p not in drop_seq]
    kept_df = train_df[train_df['pdb_id'].isin(keep)].reset_index(drop=True)

    stats = {
        'n_train': n_train,
        'n_cdhit': n_cdhit,
        'n_drop_seq': len(drop_seq),
        'n_keep': len(kept_df),
    }
    return kept_df, stats
