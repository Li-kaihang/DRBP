#!/usr/bin/env python3
"""单链版 BioLiP 结合位点标签构建 (GraphBind 口径)。

与旧版 /root/DRBP/binding_site/build_binding_site_labels.py 的关键区别：
  旧版: 按 pdb_id 聚合，parse_pdb_backbone 把同一 PDB 的所有链拼接成一条序列，
        一个 pdb_id = 一个"蛋白"（多链被拼成假连续序列，与 GraphBind 单链不符）。
  本版: 按 (pdb_id, chain) 聚合，每条链 = 一个独立蛋白（与 GraphBind 的
        DNA-573/RNA-495、以及标准测试集 DNA-129/RNA-117/DNA-181 一致），
        位点只取该链自身的标注，sequence 为该链 backbone 序列。

输入:
  - /root/DRBP/data/binding_site/biolip_dna_rna_sites.csv  (BioLiP 位点, 每行一个 (pdb,chain,ligand))
  - /root/DRBP/data/binding_site/pdb/{pdb_id}.pdb          (完整 PDB 结构)

输出:
  - data/singlechain_labels.csv     列: pdb_id, sequence, dna_positions, rna_positions
      (pdb_id = "{pdb_id}_{chain}", 如 1a02_F；dna/rna_positions 为 0-indexed)
  - data/train_pdbs/{pdb_id}_{chain}.pdb   单链 PDB (只保留该链 ATOM 行)

用法: cd /root/DRBP/finetune_6layers/binding_site_singlechain && \
      /root/.conda/envs/drbp/bin/python scripts/build_singlechain_labels.py
"""
import os, sys, csv, argparse
from collections import defaultdict

sys.path.insert(0, '/root/DRBP/finetune_6layers/final_binding_site')
from data_processing import parse_pdb_backbone, AA_3TO1

BIOLIP = '/root/DRBP/data/binding_site/biolip_dna_rna_sites.csv'
PDB_DIR = '/root/DRBP/data/binding_site/pdb'
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, 'data')
OUT_LABELS = os.path.join(DATA, 'singlechain_labels.csv')
OUT_PDB = '/tmp/bsv4/train_pdbs'   # 单链 PDB 写 /tmp 快盘 (/root 是慢 FUSE)

AA_SET = set('ACDEFGHIKLMNPQRSTVWY')


def parse_binding_residues(s):
    """'I86 Q90 R209' -> [('I', '86'), ('Q', '90'), ('R', '209')]"""
    out = []
    for tok in (s or '').split():
        tok = tok.strip()
        if not tok:
            continue
        letter = tok[0].upper()
        num = tok[1:].strip()
        if letter in AA_SET and num.isdigit():
            out.append((letter, num))
    return out


def extract_chain_pdb(lines, chain, out_path):
    """从内存里的完整 PDB ATOM 行里，把 chain 这一条链写进 out_path (单链 PDB)。

    只保留 ATOM、altloc 为 ' ' 或 'A'、chain 匹配的行，跳过 HETATM/水/配体，
    与标准测试集提链逻辑 (build_testset.py) 完全一致。入参 lines 是一次性读入的
    完整 PDB 行列表，避免 /root 慢 FUSE 上每条链重复读盘。
    """
    n = 0
    with open(out_path, 'w') as o:
        for line in lines:
            if line.startswith('ENDMDL'):
                break
            if not line.startswith('ATOM '):
                continue
            if line[16] not in (' ', 'A'):
                continue
            if line[21] == chain:
                o.write(line)
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--biolip', default=BIOLIP)
    ap.add_argument('--pdb_dir', default=PDB_DIR)
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 个 PDB (调试用)')
    args = ap.parse_args()

    os.makedirs(OUT_PDB, exist_ok=True)

    # 1. 读 BioLiP, 按 (pdb_id, chain) 聚合: {(pdb_id, chain): {dna:set, rna:set}}
    sites = defaultdict(lambda: {'dna': set(), 'rna': set()})
    # 同时记录每 (pdb_id, chain) 的 BioLiP 序列 (用于多字符链的 fallback 匹配)
    biolip_seq = {}
    pdb_ligands = defaultdict(set)   # pdb_id -> 该 PDB 出现过的核酸 ligand 集合
    with open(args.biolip) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            uniprot, pdb_id, chain, ligand, bres, seq_len, sequence = row
            if ligand not in ('dna', 'rna'):
                continue  # 忽略列错位产生的垃圾 ligand (A/C/2fwt/3vnv...)
            pdb_id = pdb_id.lower()
            biolip_seq[(pdb_id, chain)] = sequence
            pdb_ligands[pdb_id].add(ligand)
            for letter, num in parse_binding_residues(bres):
                sites[(pdb_id, chain)][ligand].add((letter, num))

    # GraphBind: 排除 DNA-RNA 混合复合物 (一个 PDB 同时结合 DNA 和 RNA 的，整条 PDB 去掉)
    mixed_pdb = {p for p, ligs in pdb_ligands.items() if 'dna' in ligs and 'rna' in ligs}
    sites = {k: v for k, v in sites.items() if k[0] not in mixed_pdb}
    print(f'DNA-RNA 混合 PDB 排除: {len(mixed_pdb)} 个', flush=True)
    print(f'共 {len(sites)} 个唯一 (pdb_id, chain) 有待标注', flush=True)

    rows = []
    n_chain = 0          # 成功写出的单链蛋白数
    n_exact = 0          # chain 单字符直接匹配
    n_bylast = 0         # 多字符 chain 取最后一个字符匹配
    n_byseq = 0          # 多字符 chain 按序列匹配
    n_skip = 0           # 未匹配上 / 无结构
    n_miss = n_mismatch = 0

    pdb_list = sorted(sites.keys())
    if args.limit > 0:
        pdb_list = pdb_list[:args.limit]

    # 按 pdb_id 分组，每个 PDB 只 parse 一次
    by_pdb = defaultdict(list)
    for (pdb_id, chain) in pdb_list:
        by_pdb[pdb_id].append(chain)

    for pdb_id, chains in by_pdb.items():
        pdb_path = os.path.join(args.pdb_dir, f'{pdb_id}.pdb')
        if not os.path.exists(pdb_path):
            for ch in chains:
                n_skip += 1
            continue
        try:
            parsed = parse_pdb_backbone(pdb_path)
        except Exception as e:
            print(f'  [skip] {pdb_id}: 解析失败 {e}', flush=True)
            for ch in chains:
                n_skip += 1
            continue
        # 完整 PDB 行一次性读入内存，供后续每条链提链复用 (避免 /root 慢盘重复读)
        with open(pdb_path) as f:
            pdb_lines = f.readlines()

        seq = parsed['sequence']
        residue_ids = parsed['residue_ids']  # [(chain_char, resseq), ...]

        # 把全局索引按 PDB 链 char 分组
        pdb_chain_idx = defaultdict(list)      # chain_char -> [global_idx...]
        for i, (ch, rs) in enumerate(residue_ids):
            pdb_chain_idx[ch].append(i)
        # 每条 PDB 链的序列 (用于 fallback)
        pdb_chain_seq = {ch: ''.join(seq[i] for i in idxs)
                         for ch, idxs in pdb_chain_idx.items()}

        for chain in chains:
            matched_chain = None
            # (a) 单字符 chain 直接匹配 PDB col21
            if len(chain) == 1 and chain in pdb_chain_idx:
                matched_chain = chain
                n_exact += 1
            # (b) 多字符 chain: 先试最后一个字符 (BioLiP 对超大结构常加前缀)
            elif len(chain) > 1 and chain[-1] in pdb_chain_idx:
                matched_chain = chain[-1]
                n_bylast += 1
            # (c) 按序列匹配
            else:
                bseq = biolip_seq.get((pdb_id, chain), '')
                best = None
                for pch, pseq in pdb_chain_seq.items():
                    if bseq and pseq == bseq:
                        best = pch
                        break
                if best is not None:
                    matched_chain = best
                    n_byseq += 1

            if matched_chain is None:
                n_skip += 1
                continue

            idxs = sorted(pdb_chain_idx[matched_chain])
            # 该链内部 (resseq -> 链内下标)
            pos_map = defaultdict(list)
            for j, gidx in enumerate(idxs):
                pos_map[residue_ids[gidx][1]].append(j)

            dna_pos, rna_pos = set(), set()
            for lig, res_set in sites[(pdb_id, chain)].items():
                for letter, num in res_set:
                    ps = pos_map.get(num, [])
                    if not ps:
                        n_miss += 1
                        continue
                    matched = [j for j in ps if seq[idxs[j]] == letter]
                    if not matched:
                        n_mismatch += 1
                        matched = ps[:1]
                    j = matched[0]
                    if lig == 'dna':
                        dna_pos.add(j)
                    else:
                        rna_pos.add(j)

            if not dna_pos and not rna_pos:
                continue  # 该链没有可映射的阳性位点

            pdb_id_chain = f'{pdb_id}_{chain}'
            chain_seq = ''.join(seq[i] for i in idxs)
            # 写单链 PDB (matched_chain 是 PDB 里真实 col21 字符)
            out_pdb = os.path.join(OUT_PDB, f'{pdb_id_chain}.pdb')
            try:
                extract_chain_pdb(pdb_lines, matched_chain, out_pdb)
            except Exception as e:
                print(f'  [skip] {pdb_id_chain}: 写单链 PDB 失败 {e}', flush=True)
                n_skip += 1
                continue

            rows.append((
                pdb_id_chain, chain_seq,
                ','.join(map(str, sorted(dna_pos))),
                ','.join(map(str, sorted(rna_pos))),
            ))
            n_chain += 1

    with open(OUT_LABELS, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pdb_id', 'sequence', 'dna_positions', 'rna_positions'])
        w.writerows(rows)

    print(f'\n=== 汇总 ===', flush=True)
    print(f'唯一 (pdb_id, chain) 输入: {len(sites)}', flush=True)
    print(f'写出单链蛋白: {n_chain}', flush=True)
    print(f'  chain 匹配: 单字符直接={n_exact}  末字符={n_bylast}  序列匹配={n_byseq}  跳过={n_skip}', flush=True)
    print(f'  位点缺失编号 {n_miss} / 字母不匹配 {n_mismatch}', flush=True)
    print(f'输出 labels: {OUT_LABELS}', flush=True)
    print(f'单链 PDB:    {OUT_PDB}/  ({n_chain} 个)', flush=True)


if __name__ == '__main__':
    main()
