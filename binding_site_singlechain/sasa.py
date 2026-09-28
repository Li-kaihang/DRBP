#!/usr/bin/env python3
"""真实 SASA 计算 (Shrake-Rupley 算法, 纯 NumPy, 无外部依赖)。

Shrake & Rupley (1973): 在每个原子表面撒点, 用 1.4Å 水探针判断点是否被埋住,
可及表面积 = 未被埋住的点比例 × 原子球面积。与 DSSP 同属"真实 SASA"(全原子+水探针)。

与旧 ca_exposure(Cα 邻居数代理) 的区别:
  - 全原子(含侧链), 不是只看 Cα
  - 连续的表面积(Å²), 不是离散的邻居计数
  - 考虑原子范德华半径 + 水探针
"""
import numpy as np

# 范德华半径 (Å), 忽略氢(重原子 SASA, 常见做法)
VDW_RADII = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'P': 1.8}
PROBE = 1.4          # 水探针半径
N_POINTS = 100       # 每个原子表面撒点数

# 20 种氨基酸的"最大可及面积"(Shrake-Rupley 归一化基准, 用于算 RSA)
# 参考 Tien et al. 2013 的 Ala-Xxx 三肽经验值
MAX_SASA = {
    'A': 121, 'R': 265, 'N': 187, 'D': 187, 'C': 148, 'Q': 214, 'E': 214,
    'G': 97, 'H': 216, 'I': 195, 'L': 191, 'K': 230, 'M': 203, 'F': 228,
    'P': 154, 'S': 143, 'T': 163, 'W': 264, 'Y': 255, 'V': 165,
}


def _fibonacci_sphere(n):
    """单位球面均匀撒 n 个点 (黄金螺旋法)"""
    pts = np.zeros((n, 3), dtype=np.float32)
    phi = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2
        r = np.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        pts[i] = [np.cos(theta) * r, y, np.sin(theta) * r]
    return pts


def parse_full_atoms(pdb_path):
    """提取标准氨基酸的全部重原子 (N,C,O,S) 坐标 + 元素 + 残基序列。

    Returns: (coords (M,3), elements (M,), res_idx (M,), seq (str))
      每个原子属于哪个残基 (res_idx), 便于聚合成逐残基 SASA。
    """
    atoms = []          # (x, y, z, element, res_key)
    res_order = []
    res_seen = set()
    aa_3to1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
               'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
               'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
               'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
    seq = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ENDMDL'):
                break
            if not line.startswith('ATOM '):
                continue
            altloc = line[16]
            if altloc not in (' ', 'A'):
                continue
            res_name = line[17:20].strip()
            if res_name not in aa_3to1:
                continue
            atom_name = line[12:16].strip()
            element = atom_name[0]           # 首字符即元素 (C/N/O/S)
            if element not in VDW_RADII:
                continue
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            chain = line[21]
            resseq = line[22:26].strip()
            res_key = (chain, resseq)
            if res_key not in res_seen:
                res_seen.add(res_key)
                res_order.append(res_key)
                seq.append(aa_3to1[res_name])
            res_i = res_order.index(res_key)
            atoms.append((x, y, z, element, res_i))
    coords = np.array([[a[0], a[1], a[2]] for a in atoms], dtype=np.float32)
    elements = [a[3] for a in atoms]
    res_idx = np.array([a[4] for a in atoms], dtype=np.int64)
    return coords, elements, res_idx, ''.join(seq)


def shrake_rupley_sasa(coords, radii):
    """计算每个原子的 SASA (Å²)。

    coords: (M,3)   radii: (M,) 范德华半径
    """
    M = coords.shape[0]
    sphere = _fibonacci_sphere(N_POINTS)          # (P, 3)
    R = radii + PROBE                              # 原子+探针球半径
    sasa = np.zeros(M, dtype=np.float32)
    # 空间分箱加速: 只检查附近的原子
    for i in range(M):
        pts = coords[i] + sphere * R[i]            # (P, 3)
        # 与其它原子的距离 (向量化)
        d = np.linalg.norm(pts[:, None, :] - coords[None, :, :], axis=-1)  # (P, M)
        # 点 j 被原子 k 埋住: 距离 < 原子k的探针球半径
        covered = (d < R[None, :])
        covered[:, i] = False                      # 排除自身
        buried = covered.any(axis=1)               # (P,) 是否被任一原子埋住
        accessible = 1.0 - buried.mean()
        sasa[i] = accessible * 4 * np.pi * R[i] ** 2
    return sasa


def compute_sasa(pdb_path):
    """逐残基 SASA + RSA。

    Returns: (abs_sasa (L,), rsa (L,))  相对可及性 rsa = abs / MAX_SASA, 夹到 [0,1]
    """
    coords, elements, res_idx, seq = parse_full_atoms(pdb_path)
    radii = np.array([VDW_RADII[e] for e in elements], dtype=np.float32)
    atom_sasa = shrake_rupley_sasa(coords, radii)
    L = len(seq)
    abs_sasa = np.zeros(L, dtype=np.float32)
    for i in range(L):
        abs_sasa[i] = atom_sasa[res_idx == i].sum()
    rsa = np.array([min(1.0, abs_sasa[i] / MAX_SASA.get(aa, 150.0)) for i, aa in enumerate(seq)],
                   dtype=np.float32)
    return abs_sasa, rsa


def compute_dssp_sasa(pdb_path):
    """用 DSSP (mkdssp) 算真实的全原子 SASA + 二级结构。

    Returns: (sasa_abs (L,), rsa (L,), ss (str 长度L))
      sasa_abs: 每残基绝对溶剂可及面积 (Å²)
      rsa:      相对可及性 [0,1] (除以该氨基酸的 MAX_SASA)
      ss:       二级结构 (DSSP 8类: H/B/E/G/I/T/S/' ')
    """
    import subprocess, os
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = '/root/miniconda3/lib:' + env.get('LD_LIBRARY_PATH', '')
    MK = '/usr/local/bin/mkdssp'
    r = subprocess.run([MK, pdb_path], capture_output=True, text=True, env=env)
    out = r.stdout
    aa_set = set('ACDEFGHIKLMNPQRSTVWY')
    sasa_list, ss_list, aa_list = [], [], []
    for line in out.split('\n'):
        if len(line) < 38:
            continue
        aa = line[13]
        if aa not in aa_set:
            continue
        acc = line[34:38].strip()
        if not acc:
            continue
        try:
            sasa = float(acc)
        except ValueError:
            continue
        aa_list.append(aa)
        sasa_list.append(sasa)
        ss_list.append(line[16] if len(line) > 16 else ' ')
    sasa_abs = np.array(sasa_list, dtype=np.float32)
    rsa = np.array([min(1.0, sasa_abs[i] / MAX_SASA.get(aa, 150.0))
                    for i, aa in enumerate(aa_list)], dtype=np.float32)
    ss = ''.join(ss_list)
    return sasa_abs, rsa, ss


if __name__ == '__main__':
    import sys
    pdb = sys.argv[1] if len(sys.argv) > 1 else '/root/DRBP/data/binding_site/pdb/10mh.pdb'
    abs_sasa, rsa = compute_sasa(pdb)
    print(f"蛋白 {pdb}: {len(abs_sasa)} 残基")
    print(f"  [Cα级代理] SASA 均值 {abs_sasa.mean():.1f}, RSA 均值 {rsa.mean():.3f}")
    d_abs, d_rsa, ss = compute_dssp_sasa(pdb)
    print(f"  [真DSSP]   SASA 均值 {d_abs.mean():.1f}, RSA 均值 {d_rsa.mean():.3f}, SS样例 {ss[:20]}")
