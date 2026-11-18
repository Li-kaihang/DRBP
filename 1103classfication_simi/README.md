# DRBP-Net — 最优架构 × 「simi」口径（train_withsimi.csv）

在**最优分类架构**（`head_scheme='two'` 双 query 头 + SharedTrunk + `--no_gate` + `drbp_upsample 5.0`
+ 解冻 ESM 最后 6 层，即 `best_twoquery.pt` 那套）上，改用 **`train_withsimi.csv`（22,181 条）** 训练。

## 目的

补全「**固定架构下**」的数据口径阶梯中间档，把泄漏来源拆干净：

| 口径 | 条数 | 泄漏特征 | two-query 最优架构 |
|---|---|---|---|
| train_final | 22,892 | 含与测试集 100% 相同 685 条 + 部分同源 | ✅ `best_twoquery.pt` |
| **train_withsimi** | **22,181** | **去 100% 相同、保留 30–99% 部分同源** | ✅ **本目录（本次）** |
| train_clean30 | 19,795 | 严格去 ≥30% 同源 | ✅ `classfication_clean30_twohead/ep02.pt` |

- `train_withsimi.csv` 定义见 `ablation_no_non/classification_4class/data_withsimi/README.md`：
  内部精确去重 + 剔除与 4 测试集「一模一样」的序列，**保留所有部分相似同源**。
- 之前 simi 只跑过 `--head_scheme four`（4 分类头，`run_esm2_four_withsimi.sh`），
  没在 two-query 最优架构上跑过 —— 这是本次补上的缺口。

## 训练命令（`run.sh`）

与 `classfication_twoquery/run.sh` 完全一致，**唯一改动** `--train_csv train_withsimi.csv` + `--tag no_non_650M_withsimi`：

```bash
./run.sh   # 4× P100, batch_size=2/accum=2, lr 1e-4, head_scheme two, drbp_upsample 5.0
```

## 产物

- 日志：`train_withsimi.log`
- 权重 / 指标 / 逐样本预测：`runs/{ts}_no_non_650M_withsimi/`（config.csv / metrics.csv / epochs/*.pt / predictions_*.csv）

## 复现

```bash
cd /root/final_DRBP/1103classfication_simi
PY=/root/.conda/envs/drbp/bin/python
./run.sh
```

环境：`/root/.conda/envs/drbp`（torch 2.2.2+cu118）；ESM 模型 `/root/DRBP/models/esm2_t33_650M_UR50D`。

---

## 训练结果（ep02，val loss 选点）

| 指标 | 值 |
|---|---|
| val_loss | 0.6425 |
| dbp_auc / rbp_auc / drbp_auc | 0.9723 / 0.9617 / 0.8823 |
| mcc | 0.8201 |
| corr_dr | -0.1493 |
| val 搜索阈值 best_t_d / best_t_r | 0.45 / 0.95 |

权重：`runs/20261102_235135_no_non_650M_withsimi/epochs/ep02.pt`

## 人类蛋白组预测（20410 条，严格 GO 参考集）

脚本 `human_proteom/predict_ep02_proteome.py`，产物 `human_proteom/proteome_predictions_ep02_withsimi.csv`。

- 严格 GO 参考集：DBP 2267 / RBP 1598；蛋白组内命中 DBP 2252 / RBP 1594。
- 干净召回（0.85/0.75，排除训练集同 accession）：DBP 84.0%（1357/1615）、RBP 80.8%（879/1088）。

### 不同阈值下的召回 / 精度 / 预测条数

| 阈值 (dbp/rbp) | DBP 召回 | DBP 精率 | DBP 预测条 | RBP 召回 | RBP 精率 | RBP 预测条 | 双结合 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.85 / 0.75 | 82.6% | 81.7% | 2278 | 80.6% | 43.1% | 2979 | 168 |
| 0.74 / 0.63 | 85.0% | 76.5% | 2504 | 84.7% | 29.6% | 4558 | 277 |
| 0.70 / 0.60 | 85.7% | 74.5% | 2590 | 85.9% | 27.5% | 4979 | 314 |
| **0.60 / 0.60** | **88.1%** | **64.0%** | **3102** | **85.9%** | **27.5%** | **4979** | **615** |
| 0.65 / 0.45 | 86.7% | 70.9% | 2753 | 88.8% | 21.5% | 6583 | 481 |
| 0.50 / 0.50 | 91.7% | 45.8% | 4507 | 88.0% | 23.0% | 6107 | 1691 |
| 0.45 / 0.95 | 93.1% | 40.4% | 5186 | 67.6% | 66.9% | 1611 | 318 |
| 0.30 / 0.30 | 95.1% | 29.6% | 7233 | 91.0% | 18.1% | 8028 | 4071 |

### 最终阈值决策：**0.60 / 0.60**

- 只追求召回（主指标）的前提下，DBP 88.1%、RBP 85.9%，两者都 ≥85%，达标。
- 代价是 RBP 精度只有 27.5%（≈把 24% 蛋白组标成 RBP，4 个里有 3 个假阳性）；DBP 精度 64.0% 尚可。
- 注意：RBP 精度在 0.60→0.63→0.75 全程都在 27–43% 区间上不去，是**模型对 RBP 区分度不足**（训练侧问题），不是阈值能救的。
- 对比方案 0.70/0.60：DBP 召回只降 2.4pp（88.1→85.7%）、精度 +10.5pp（64.0→74.5%），RBP 完全不变——若后续要在「召回≥85%」前提下少放 DBP 假阳性，可改用 0.70/0.60。本次先按用户口径统一 **0.60/0.60** 记录。

## 后续计划

1. **不再训练**——以 `train_withsimi`（ep02）为最终 simi 口径结果。
2. **相似性去除测试**：对测试集/参考集做与训练集的同源去除，评测 `ep02.pt` 在**非同源**样本上的表现，看真实泛化能力（现有干净召回 DBP 84.0% / RBP 80.8% 只是「排除同 accession」，未按相似度序列级去重）。
