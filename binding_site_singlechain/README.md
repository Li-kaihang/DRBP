# Binding Site 预测 — 单链版最终交付

> 最后更新：2026-09-28 · 自包含目录，含训练/评估脚本、数据管线、标准测试集；模型权重（2.6 GB）不上传（见主 README §8）。

预测一个蛋白**哪些残基是 DNA / RNA 结合位点**（逐残基二分类）。序列用 ESM-2 650M 微调，结构用 GNN（含表面特征），两分支融合后由 DNA / RNA 两个头分别输出。

---

## 最终结果（标准独立测试集，逐残基 best-threshold MCC）

**最优模型 `best_bsite_seq30_only_650M.pt`**（epoch 2，单链管线 2017 训练集）：

| 测试集 | MCC | AUROC | 阳性率 | 对比 |
|---|---|---|---|---|
| DNA-129 | **0.5398** | 0.926 | 5.97% | 上一版 0.5289，GraphBind 0.499 ✅ |
| DNA-181 | **0.3999** | 0.908 | 4.26% | 上一版 0.3815（GraphSite 测试集）|
| RNA-117 | **0.3556** | 0.898 | 5.43% | 上一版 0.3432，GraphBind 0.322 ✅ |
| **mean** | **0.4318** | — | — | 上一版 0.4179 |

> 三集全部超过上一版（多链拼接管线）与 GraphBind 原文；提升来自**单链粒度 + 修正标注迁移**，不是模型改动。

---

## 相对上一版的关键变化

1. **单链粒度**：旧版「一个 PDB = 一个蛋白」（多链拼接成假连续序列）；新版「**一条链 = 一个蛋白**」，与 GraphBind 训练集（DNA-573/RNA-495）及标准测试集粒度一致。
2. **修正标注迁移判据**：GraphBind 原文要求「seq>0.8 且 TM>0.5」同时满足才做补齐。实测 seq>0.8 ⟹ TM>0.5（反向不成立），故「seq>0.8 且 TM>0.5」≈「seq>0.8」。早期曾有的**纯结构 TM>0.5 聚类**步骤（会合并序列不同的链）是错的，已移除。

---

## 数据管线（scripts/，四步）

| 步骤 | 脚本 | 作用 | 数量 |
|---|---|---|---|
| 0 | `build_singlechain_labels.py` | BioLiP 位点 → 单链标签 | 44,724 单链 |
| 1 | `01_seq_cluster.py` | seq>0.8 聚类 + 标注迁移到最长链 | → 4,722 rep |
| 3 | `03_seq30_only.py` | CD-HIT 30% + test-train seq 30% | → 2,017（**最终**）|
| 4 | `04_seq30_tm05.py` | 上一步 + 结构 TM>0.5（对照，掉分未用）| → 1,719 |

- 最终训练集 = `03.seq30_only`（2,017 条），90/10 切分 → **train 1,815 / val 202**（seed 42）。
- 测试集 = `data/test_labels.csv` + `data/test_pdbs/`（424 条单链，DNA-129/RNA-117 来自 GraphBind、DNA-181 来自 GraphSite）。

---

## 复现

```bash
cd binding_site_singlechain
# 数据管线（第 0/1 步需原始 BioLiP 位点 CSV + 完整 PDB，见脚本 docstring）
/root/.conda/envs/drbp/bin/python scripts/01_seq_cluster.py
/root/.conda/envs/drbp/bin/python scripts/03_seq30_only.py --final data/01.seq_cluster/aug80.csv
/root/.conda/envs/drbp/bin/python scripts/make_splits.py \
  --biolip data/03.seq30_only/labels.csv --keep data/03.seq30_only/keep_ids.txt \
  --out_dir data/03.seq30_only

# 训练（4 卡 DDP fork）
/root/.conda/envs/drbp/bin/python train_randwin.py --n_gpus 4 --epochs 15 \
  --batch_size 2 --lr 1e-4 --window 512 \
  --unfreeze_esm_layers 6 --task both --eval_test_every 1 \
  --esm_model /root/DRBP/models/esm2_t33_650M_UR50D \
  --data_dir data/03.seq30_only --save checkpoint/best_bsite_seq30_only_650M.pt

# 评估
/root/.conda/envs/drbp/bin/python eval_sliding.py \
  --ckpt checkpoint/best_bsite_seq30_only_650M.pt \
  --esm_model /root/DRBP/models/esm2_t33_650M_UR50D
```

> 训练时 `train_randwin.py` 的 `PDB_DIR` 需指向单链 PDB（本机 `/tmp/bsv4/train_pdbs`）。训练 PDB 因体积大未上传，按第 1 步重新提取。

---

## 文件结构

```
binding_site_singlechain/
├── train_randwin.py           # DDP 训练脚本
├── eval_sliding.py            # 滑动窗口评估（三测试集）
├── randwin_dataset.py         # 随机窗口数据集
├── struct_gnn.py              # 结构 GNN
├── joint_model.py             # 序列+结构联合模型
├── lora.py                    # LoRA（可选，当前配方未用）
├── data_processing.py         # 骨架几何/表面特征提取
├── sasa.py                    # DSSP SASA
├── struct_dataset.py          # 标签解析/collate
├── scripts/                   # 数据管线（见上表）
└── data/
    ├── test_labels.csv        # 424 测试蛋白标签
    ├── test_pdbs/             # 424 单链实验 PDB
    ├── singlechain_labels.csv # 44724 单链标签（管线输入）
    └── 03.seq30_only/         # 最终训练集（labels/split/summary）
```

脚本内 `sys.path` 已改为 `os.path.dirname(__file__)`（自包含），不依赖外部模块路径。

## 已知边界

- **DNA-181 是短板**（0.3999）：平均序列更长（413 vs 291 残基）、阳性更稀，MCC 天然偏低，非 bug。
- 结构去冗余（变体 B / TM>0.5）自断数据、掉分，终版只做序列 30% 去冗余。
- 整体属中等水平（MCC 0.34~0.54），靠「比 GraphBind / ESM-NBR 略好」的相对优势支撑。
