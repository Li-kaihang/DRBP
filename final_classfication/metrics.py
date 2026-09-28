"""
指标计算 / 阈值搜索 / 塌缩诊断
==============================
与旧 train_baseline.py 的 compute_metrics 相比有三点不同:

1. 阈值可传入 (t_d, t_r), 不再硬编码 0.5
2. 第三个头是 DRBP (不是 non), 直接参与指标
3. 新增塌缩诊断 + DRBP 的 precision/F1

关于 precision: 旧指标只报 drbp_caught/drbp_total (召回率), 一个对所有输入都
喊 "DRBP" 的退化模型能拿到 100% 召回。必须同时看 precision 才知道是不是真的。
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef

# 4 类编码: 0=non, 1=DBP-only, 2=RBP-only, 3=DRBP
CLASS_NAMES = ['non', 'dbp', 'rbp', 'drbp']


def _auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')


def to_class(dbp_flag, rbp_flag):
    """(bool, bool) 数组 → 4 类整数编码"""
    c = np.zeros(len(dbp_flag), dtype=int)
    c[dbp_flag & ~rbp_flag] = 1
    c[~dbp_flag & rbp_flag] = 2
    c[dbp_flag & rbp_flag] = 3
    return c


def drbp_decision(dbp_p, rbp_p, drbp_p, t_d, t_r, t_b, use_drbp_head=True):
    """
    DRBP 判定。

    use_drbp_head=True (drbp 方案):
      (drbp_p > t_b) OR (dbp_p > t_d AND rbp_p > t_r)
      DRBP 头是唯一被 y_d·y_r 直接监督的头, 让它直接参与判定, 主头双阳作补充。

    use_drbp_head=False (non 方案):
      单纯 (dbp_p > t_d AND rbp_p > t_r)。DRBP 没有独立头, 由两个主头共同开火派生。
    """
    if not use_drbp_head:
        return (dbp_p > t_d) & (rbp_p > t_r)
    return (drbp_p > t_b) | ((dbp_p > t_d) & (rbp_p > t_r))


def compute_metrics(dbp_p, rbp_p, drbp_p, dbp_y, rbp_y, t_d=0.5, t_r=0.5, t_b=0.5,
                    use_drbp_head=True):
    """
    dbp_p/rbp_p/drbp_p : (N,) 概率
    dbp_y/rbp_y        : (N,) 0/1 标签
    t_d/t_r/t_b        : 判正阈值
    use_drbp_head      : False = non 方案 (DRBP 由 DBP∧RBP 派生, 无独立头)
    """
    dbp_y = dbp_y.astype(int)
    rbp_y = rbp_y.astype(int)
    both_y = (dbp_y & rbp_y)

    pred = to_class(dbp_p > t_d, rbp_p > t_r)
    pred[drbp_decision(dbp_p, rbp_p, drbp_p, t_d, t_r, t_b, use_drbp_head)] = 3
    true = to_class(dbp_y == 1, rbp_y == 1)

    pred_drbp = (pred == 3)
    true_drbp = (true == 3)
    tp = int((pred_drbp & true_drbp).sum())
    n_pred = int(pred_drbp.sum())
    n_true = int(true_drbp.sum())
    prec = tp / n_pred if n_pred else float('nan')
    rec = tp / n_true if n_true else float('nan')
    f1 = 2 * prec * rec / (prec + rec) if n_pred and n_true and (prec + rec) > 0 else float('nan')

    m = {
        'dbp_auc': _auc(dbp_y, dbp_p),
        'rbp_auc': _auc(rbp_y, rbp_p),
        'drbp_auc': _auc(both_y, drbp_p),          # 用 DRBP 头自己的分数, 不再是 min()
        'acc': accuracy_score(true, pred),
        'mcc': matthews_corrcoef(true, pred),
        'drbp_caught': tp,
        'drbp_total': n_true,
        'drbp_pred': n_pred,                        # 预测成 DRBP 的总数 —— 揭穿"全喊yes"
        'drbp_prec': prec,
        'drbp_rec': rec,
        'drbp_f1': f1,
        't_d': t_d,
        't_r': t_r,
        't_b': t_b,
    }
    # 各类召回, 看是不是某一类被完全放弃
    for k, name in enumerate(CLASS_NAMES):
        sel = (true == k)
        m[f'rec_{name}'] = float((pred[sel] == k).mean()) if sel.any() else float('nan')
    m['macro_f1'] = macro_f1(true, pred)

    # 真 DRBP 上两个主头同时开火的比例 —— 直接监控"只有一个头开火"这个病。
    # 实测 (DRBP206, ep3): 103 个真 DRBP 里 DBP 头只开火 42 个 (中位概率 0.382),
    # RBP 头开火 74 个 (中位 0.878), 两个同时开火只有 21 个。若共现损失起作用,
    # 这个数应该明显上升。纯诊断列, 不参与任何已有指标的计算。
    if n_true:
        sel = (both_y == 1)
        m['coact'] = float(((dbp_p > t_d) & (rbp_p > t_r))[sel].mean())
        m['med_p_weak'] = float(np.median(np.minimum(dbp_p, rbp_p)[sel]))
    else:
        m['coact'] = float('nan')
        m['med_p_weak'] = float('nan')
    return m


def macro_f1(true, pred):
    """四类 F1 的宏平均 —— 每个类等权, 防止任何一类单方面塌掉"""
    fs = []
    for k in range(4):
        tp = int(((pred == k) & (true == k)).sum())
        n_pred = int((pred == k).sum())
        n_true = int((true == k).sum())
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_true if n_true else 0.0
        fs.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(fs))


def collapse_diagnostics(dbp_p, rbp_p, drbp_p):
    """
    塌缩诊断。旧模型三头两两相关系数是 0.998~1.000, 98.5% 样本 >0.5。
    这几个数在第一个 epoch 就能暴露问题, 不用跑完再回头查。
    """
    def corr(a, b):
        if a.std() < 1e-8 or b.std() < 1e-8:
            return float('nan')          # 常数输出本身就是塌缩
        return float(np.corrcoef(a, b)[0, 1])

    d = {
        'corr_dr': corr(dbp_p, rbp_p),
        'corr_db': corr(dbp_p, drbp_p),
        'corr_rb': corr(rbp_p, drbp_p),
        'frac_d_pos': float((dbp_p > 0.5).mean()),
        'frac_r_pos': float((rbp_p > 0.5).mean()),
        'frac_b_pos': float((drbp_p > 0.5).mean()),
        'std_d': float(dbp_p.std()),
        'std_r': float(rbp_p.std()),
        'std_b': float(drbp_p.std()),
        # 层级一致性: DRBP 头 vs 两个主头的乘积, 应该趋近 0
        'consistency_mse': float(((drbp_p - dbp_p * rbp_p) ** 2).mean()),
    }
    pred = to_class(dbp_p > 0.5, rbp_p > 0.5)
    for k, name in enumerate(CLASS_NAMES):
        d[f'pred_{name}'] = int((pred == k).sum())
    return d


def search_thresholds(dbp_p, rbp_p, drbp_p, dbp_y, rbp_y, lo=0.05, hi=0.95, step=0.05,
                      use_drbp_head=True):
    """
    在 val 上分两步搜 (t_d, t_r, t_b):

    1. (t_d, t_r) 按 4 类 MCC 搜 —— non/DBP/RBP 由主头决定
    2. (t_b)     **单独**按 DRBP 的 F1 搜 —— 不再被 non 类绑架 (仅 drbp 方案)

    为什么 t_b 必须单独搜: val 里 DRBP 只占 1.7% (39/2290)。如果 t_b 也塞进 4 类 MCC
    一起最大化, 最优解就是把 t_b 抬到 0.9、宁可漏光 DRBP —— 因为靠 non/DBP/RBP 三类就
    能把整体 MCC 撑高, 稀有的 DRBP 类成了牺牲品。DRBP 头其实学得很好 (t_b=0.5 就能
    58/103、94% 精确率), 是搜索目标错了。

    use_drbp_head=False (non 方案): 没有 DRBP 头, t_b 无意义, 固定 0.5, 只搜 (t_d, t_r)。
    """
    dbp_y = dbp_y.astype(int)
    rbp_y = rbp_y.astype(int)
    true = to_class(dbp_y == 1, rbp_y == 1)
    grid = np.arange(lo, hi + 1e-9, step)

    # 1. 主头阈值: 4 类 MCC (DRBP 头不参与, 用主头双阳近似 DRBP)
    best_m = (0.5, 0.5, -2.0)
    for t_d in grid:
        d_flag = dbp_p > t_d
        for t_r in grid:
            mcc = matthews_corrcoef(true, to_class(d_flag, rbp_p > t_r))
            if mcc > best_m[2]:
                best_m = (float(t_d), float(t_r), float(mcc))
    t_d, t_r, _ = best_m

    if not use_drbp_head:
        return t_d, t_r, 0.5, best_m[2]

    # 2. DRBP 头阈值: DRBP 的 F1 (在 val 的 DRBP 样本上)
    both_y = (dbp_y & rbp_y).astype(int)
    best_b = (0.5, -1.0)
    for t_b in grid:
        pred = (drbp_p > t_b).astype(int)
        tp = int((pred & both_y).sum())
        n_pred = int(pred.sum())
        n_true = int(both_y.sum())
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_true if n_true else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_b[1]:
            best_b = (float(t_b), float(f1))
    t_b, _ = best_b
    return t_d, t_r, t_b, best_m[2]
