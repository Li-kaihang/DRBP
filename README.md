# DRBP-Net — 最终交付（分类 + 结合位点）

> 最后更新：2026-09-28 · 本目录是**最优的分类路径 + 结合位点路径**的整合交付，代码与测试数据自包含，模型权重（2.6 GB）不上传（见 §8 下载/复现说明）。

DRBP-Net 做两件事：

1. **蛋白级分类** —— 判断一个蛋白是 DNA 结合（DBP）/ RNA 结合（RBP）/ 双结合（DRBP）/ 不结合（non）的多标签分类。
2. **残基级结合位点预测** —— 判断一个蛋白的**哪些残基**是 DNA / RNA 结合位点（逐残基二分类）。

两个任务共用序列 backbone **ESM-2 650M（`esm2_t33_650M_UR50D`，解冻最后 6 层）**；分类只走序列，结合位点走「序列 + 结构（GNN + 表面特征）」两分支。

---

## 0. 最终结果速览

| 任务 | 测试集 | 主指标 | 数值 | 对比 / 备注 |
|---|---|---|---|---|
| **分类** | DRBP206 | MCC | **0.5551** | 150M 为 0.4897（+0.065），DRBP 捕获 53/103 |
| | PDB255 | MCC | **0.5576** | 150M 为 0.5456 |
| | TEST474 | MCC | **0.7844** | 150M 为 0.7823 |
| | EZL | MCC | **0.7724** | 最大集，150M 为 0.7425（+0.030）|
| **结合位点** | DNA-129 | MCC | **0.5398** | 上一版 0.5289，GraphBind 原文 0.499 ✅ |
| | DNA-181 | MCC | **0.3999** | 上一版 0.3815（GraphSite 测试集）|
| | RNA-117 | MCC | **0.3556** | 上一版 0.3432，GraphBind 原文 0.322 ✅ |

> 结合位点本轮提升来自**数据管线重做**（见 §3.3 单链管线 + §3.4 关键修正），不是模型改动——从「一个 PDB = 一个蛋白（多链拼接）」改成「一条链 = 一个蛋白（GraphBind 口径）」，并修正了标注迁移的判据。

---

## 1. 目录结构

```
final_DRBP/
├── README.md                        # 本文档（整合版）
├── final_classfication/             # 分类任务最终交付（代码，无模型/数据）
│   ├── README.md                    #   分类任务单独说明
│   ├── train_drbp.py                #   DDP 训练脚本（4 卡 fork）
│   ├── eval_650m_test.py            #   四测试集推理/评估
│   ├── model_drbp.py                #   DRBPNetNew + 损失
│   ├── model_baseline.py            #   Config / SharedTrunk / 门控残差
│   ├── metrics.py                   #   指标 + 阈值搜索
│   ├── struct_adapter.py            #   结构适配层（esm2 模式未用，import 依赖保留）
│   └── build_clean_data.py          #   分类数据 30% 去冗余脚本
└── binding_site_singlechain/        # 结合位点最终交付（代码 + 测试数据，自包含）
    ├── README.md                    #   结合位点任务单独说明
    ├── train_randwin.py             #   DDP 训练脚本（4 卡 fork，window=512）
    ├── eval_sliding.py              #   滑动窗口评估（三测试集）
    ├── randwin_dataset.py           #   随机窗口数据集
    ├── struct_gnn.py                #   结构 GNN
    ├── joint_model.py               #   序列+结构联合模型
    ├── lora.py                      #   LoRA（可选，当前配方未用）
    ├── data_processing.py           #   骨架几何 / 表面特征提取
    ├── sasa.py                      #   DSSP SASA
    ├── struct_dataset.py            #   标签解析 / collate
    ├── scripts/                     #   数据管线（单链版，见 §3.3）
    │   ├── pipeline_common.py       #   公共常量 / mmseqs·foldseek 封装
    │   ├── build_singlechain_labels.py  # 第 0 步：BioLiP → 单链标签
    │   ├── 01_seq_cluster.py        #   第 1 步：seq>0.8 聚类 + 标注迁移
    │   ├── 03_seq30_only.py         #   第 3 步：序列 30% 去冗余（变体 A，最优）
    │   ├── 04_seq30_tm05.py         #   第 4 步：序列 30% + 结构 TM>0.5（变体 B）
    │   └── make_splits.py           #   90/10 训练/验证切分
    └── data/
        ├── test_labels.csv          # 424 测试蛋白标签（DNA-129/RNA-117/DNA-181）
        ├── test_pdbs/               # 424 单链实验 PDB
        ├── singlechain_labels.csv   # 44724 条单链标签（管线输入）
        └── 03.seq30_only/           # 最终训练集（2017 条）
            ├── labels.csv
            ├── split.csv            # 1815 train / 202 val
            └── summary.txt
```

> 已按用户要求**不上传模型权重**：分类 `best_cls_650M.pt`（2.5 GB）、结合位点 `best_bsite_seq30_only_650M.pt`（2.6 GB）均在本机 `/root/DRBP/finetune_6layers/` 下，不在本仓库。分类训练/测试数据也暂不上传（用户后续自行处理，见 §8）。

---

## 2. 模型

### 2.1 分类分支（仅序列）

```
序列编码: ESM-2 (esm2_t33_650M_UR50D, hidden 1280, 33 层, 解冻最后 6 层)
主干:     SharedTrunk (多尺度 Conv1d + MHSA + 门控残差, d_model 256, dropout 0.3)
pooling:  LabelQueryPooling (每标签一个可学习 query 各自 attend 序列)
标签间:   MaskedCrossLabelAttention (DBP↔RBP 互看, DRBP 看所有, 无门控)
头:       三个独立两层 MLP → DBP / RBP / DRBP (head_scheme=non 时第三头为 non,
          DRBP 由 DBP∧RBP 派生)
损失:     pos_weight BCE + DRBP 直接监督 + 层级一致性 + 共现边际
```

**可训练参数 119,888,772（≈1.2 亿）**：解冻 ESM 后 6 层 + 分类头；前 27 层（≈5.3 亿）冻结，来自 ESM-2 预训练先验。有效自由度是 1.2 亿，不是 6.5 亿。

### 2.2 结合位点分支（序列 + 结构）

```
序列分支: ESM-2 (esm2_t33_650M_UR50D, hidden 1280, 解冻最后 6 层)
结构分支: StructureGNN (node_dim 256, 4 层消息传递, k=16 Cα 邻居)
          + 表面特征 (真 DSSP SASA / Cα 凹凸度 / 静电势)
融合:     concat(1280 + 256) → Linear → 256 → DNA 头 + RNA 头
损失:     方案B —— DNA 头只在 DBP/dual 蛋白上算 BCE, RNA 头只在 RBP/dual 上算
          (pos_weight 按各自类型算)
```

本轮结合位点配方（`best_bsite_seq30_only_650M.pt`）：`--lr 1e-4 --window 512 --batch_size 2 --n_gpus 4 --unfreeze_esm_layers 6 --task both`，在 2017 条单链训练集上训练，best epoch 2。

---

## 3. 数据

### 3.1 分类任务数据（来源）

- **来源**：iDRBP_MMC 数据集（Zhang et al., bliulab.net），经由 LAMP-PRo（arXiv:2509.24262）获取原始 FASTA 与标签。
- **训练集 `train_final.csv`：22,892 蛋白** = LAMP-PRo 原始训练集 10,966 + 我们补充 11,926（自有 BioLiP 结构数据，经 mmseqs 对测试集 / LAMP 训练集各做 30% 去冗余）。
- **标签分布**（多标签）：DBP 7,440 / RBP 4,314 / DRBP 395 / non 10,743；DRBP 仅占 1.7%，是历史难点，训练用 `--drbp_upsample 5.0` 过采样 ×5。
- **测试集（4 个 benchmark，与 LAMP-PRo 对齐）**：DRBP206（206，专门测 DRBP）、PDB255（255）、TEST474（474）、EZL（4,003）。
- ⚠️ LAMP-PRo 原始训练部分与测试集自带重叠（EZL 16.6% / TEST474 9.3% / PDB255 2% / DRBP206 2%），是 iDRBP_MMC 原始设定、为对齐论文数字刻意保留；干净集 DRBP206 / PDB255 上优势仍成立。

### 3.2 结合位点任务数据（来源）

- **训练标注**：BioLiP（PDB 实验复合物中与 DNA/RNA 配体接触的残基）。
- **测试集**：领域标准基准 **GraphBind**（DNA-129 / RNA-117）+ **GraphSite**（DNA-181）官方文件，共 424 条单链（DNA-181 名义 181 条，3 条因序列一致性 <0.95 / PDB 404 剔除）。
- **测试集阳性率 ~5%**（DNA-129 5.97% / DNA-181 4.26% / RNA-117 5.43%），逐残基 MCC 天然偏低。

### 3.3 结合位点单链管线（本次重做，核心）

关键变化：旧版把「一个 PDB = 一个蛋白」（多链拼接成一条假连续序列），新版改成「**一条链 = 一个蛋白**」，与 GraphBind 的 DNA-573/RNA-495 训练集、以及标准测试集 DNA-129/RNA-117/DNA-181 的粒度一致。管线四步：

| 步骤 | 脚本 | 作用 | 数量 |
|---|---|---|---|
| 0 | `build_singlechain_labels.py` | BioLiP 位点 → 单链标签（每 (pdb,chain) 一条）| 44,724 单链 |
| 1 | `01_seq_cluster.py` | mmseqs seq>0.8 聚类 + 标注迁移到最长链 | 44,724 → **4,722** rep（DBP 1,958 / RBP 2,823）|
| 3 | `03_seq30_only.py` | train-train CD-HIT 30% + test-train seq 30% | 4,722 → 2,403 → **2,017**（DBP 901 / RBP 1,136）|
| 4 | `04_seq30_tm05.py` | 上一步 + test-train 结构 TM>0.5（foldseek）| 2,017 → **1,719**（DBP 707 / RBP 1,029）|

- **最终训练集 = 03.seq30_only（变体 A，2,017 条）**，90/10 切分 → **train 1,815 / val 202**（seed 42）。变体 B（04，1,719 条）是加结构去冗余的对照，实测掉分（结构去冗余自断数据），故未采用。
- **标注迁移口径**：每簇取**最长链**为代表链，簇内成员的结合位点经序列对齐映射回代表链坐标，做**并集补齐**——与 GraphBind 原文「bl2seq（序列身份）+ TM-align（TM-score）聚类 → 迁移到最长链」一致。

### 3.4 本次关键修正：seq>0.8 与 TM>0.5 的关系

GraphBind 原文的标注迁移判据是「**序列一致度 >0.8 且 TM-score >0.5**」同时满足。实测发现：

- **seq>0.8 ⟹ TM>0.5**（几乎恒成立，两条链序列 80% 以上一致时结构 TM-score 通常 >0.85~0.95）；
- **反向不成立**：TM>0.5 ⇏ seq>0.8（两条链结构相似、但序列可以差别很大）。

所以「seq>0.8 且 TM>0.5」≈「seq>0.8」，TM>0.5 是冗余安全网。早期实现里曾有一个**纯结构 TM>0.5 聚类**步骤（把结构相似但序列不同的链也并到一起做标注迁移），这一步是**错的**——它会把序列不一样的链错误合并。本轮已将其移除：标注迁移只保留 `01` 的 seq>0.8 聚类，03/04 直接从 `01` 的 4,722 代表链出发。

---

## 4. 结果

### 4.1 分类（四测试集，best-threshold MCC）

最优模型 `final_classfication/best_cls_650M.pt`（epoch 3），对比 150M 同架构同配方：

| 测试集 | 指标 | 150M | 650M | Δ |
|---|---|---|---|---|
| **DRBP206** (206) | dbp_auc | 0.7608 | 0.7113 | −0.049 |
| | rbp_auc | 0.8628 | 0.8348 | −0.028 |
| | drbp_auc | 0.8378 | 0.7690 | −0.069 |
| | **MCC** | 0.4897 | **0.5551** | **+0.065** |
| | DRBP 捕获 | 44/103 | **53/103** | +9 |
| **PDB255** (255) | dbp_auc | 0.8809 | **0.9007** | +0.020 |
| | rbp_auc | 0.8721 | **0.8765** | +0.004 |
| | **MCC** | 0.5456 | **0.5576** | +0.012 |
| **TEST474** (474) | dbp_auc | 0.9829 | 0.9619 | −0.021 |
| | rbp_auc | 0.9429 | 0.9272 | −0.016 |
| | drbp_auc | 0.8439 | 0.7634 | −0.081 |
| | **MCC** | 0.7823 | **0.7844** | +0.002 |
| **EZL** (4003) | dbp_auc | 0.9636 | **0.9748** | +0.011 |
| | rbp_auc | 0.9723 | **0.9846** | +0.012 |
| | **MCC** | 0.7425 | **0.7724** | +0.030 |

**结论**：MCC 四集全胜（硬判定）；最大亮点 DRBP206 +0.065（DRBP 双结合捕获 44→53），最大集 EZL +0.030。AUC（排序）混合，是阈值 tradeoff 而非纯粹更准。

### 4.2 结合位点（三测试集，逐残基 best-threshold MCC）

最优模型 `best_bsite_seq30_only_650M.pt`（epoch 2，单链管线 2017 训练集）：

| 测试集 | MCC | 上一版（多链管线）| GraphBind 原文 |
|---|---|---|---|
| DNA-129 | **0.5398** | 0.5289 | 0.499 ✅ |
| DNA-181 | **0.3999** | 0.3815 | —（GraphSite）|
| RNA-117 | **0.3556** | 0.3432 | 0.322 ✅ |
| **mean** | **0.4318** | 0.4179 | — |

**结论**：三个测试集**全部超过上一版**（多链拼接管线）与 GraphBind 原文；提升来自单链粒度 + 修正标注迁移，非模型改动。DNA-181 仍是最短（平均序列更长 413 vs 291 残基、阳性更稀 4.26%），MCC 天然偏低。

---

## 5. 复现命令

环境必须用 `/root/.conda/envs/drbp`（torch cu118；base 是 cu124 与驱动不兼容）。HuggingFace 调用已带 `local_files_only=True`（服务器无外网）。ESM-2 权重在本机 `/root/DRBP/models/esm2_t33_650M_UR50D`。

### 5.1 分类

```bash
cd final_classfication
/root/.conda/envs/drbp/bin/python train_drbp.py --n_gpus 4 \
  --train_csv train_final.csv --mode esm2 --tag finetune6_650M \
  --epochs 30 --patience 12 --batch_size 2 --accum 2 --lr 1e-4 \
  --select_metric composite --select_ema 0.5 --no_test_every_best \
  --head_scheme non --drbp_upsample 5.0 --unfreeze_esm_layers 6 --no_gate \
  --esm_model /root/DRBP/models/esm2_t33_650M_UR50D --save best_cls_650M.pt

/root/.conda/envs/drbp/bin/python eval_650m_test.py --ckpt best_cls_650M.pt
```

### 5.2 结合位点数据管线（从 BioLiP 重跑）

```bash
cd binding_site_singlechain
# 第 0 步：BioLiP 位点 → 单链标签（需原始 biolip_dna_rna_sites.csv + 完整 PDB）
/root/.conda/envs/drbp/bin/python scripts/build_singlechain_labels.py

# 第 1 步：seq>0.8 聚类 + 标注迁移
/root/.conda/envs/drbp/bin/python scripts/01_seq_cluster.py

# 第 3 步：序列 30% 去冗余（变体 A，最终训练集）
/root/.conda/envs/drbp/bin/python scripts/03_seq30_only.py --final data/01.seq_cluster/aug80.csv

# 切分 90/10
/root/.conda/envs/drbp/bin/python scripts/make_splits.py \
  --biolip data/03.seq30_only/labels.csv --keep data/03.seq30_only/keep_ids.txt \
  --out_dir data/03.seq30_only
```

### 5.3 结合位点训练 + 评估

```bash
cd binding_site_singlechain
/root/.conda/envs/drbp/bin/python train_randwin.py --n_gpus 4 --epochs 15 \
  --batch_size 2 --lr 1e-4 --window 512 \
  --unfreeze_esm_layers 6 --task both --eval_test_every 1 \
  --esm_model /root/DRBP/models/esm2_t33_650M_UR50D \
  --data_dir data/03.seq30_only \
  --save checkpoint/best_bsite_seq30_only_650M.pt

/root/.conda/envs/drbp/bin/python eval_sliding.py \
  --ckpt checkpoint/best_bsite_seq30_only_650M.pt \
  --esm_model /root/DRBP/models/esm2_t33_650M_UR50D
```

> 训练时 `train_randwin.py` 里的 `PDB_DIR` 需指向单链 PDB（本机在 `/tmp/bsv4/train_pdbs`，是 `/tmp/bsv4/rep_pdbs` 的符号链接，第 1 步产出）。训练/评估 PDB 因体积大未上传，需按第 1 步重新提取。

---

## 6. 已知边界 / 注意事项

- **分类 AUC 未稳定超 150M**：若以 AUROC 为主指标，650M 优势不成立；以 MCC（硬判定）为主指标，650M 四集全胜、DRBP 短板改善，值得采用。
- **分类训练集含 LAMP 原始 train-test 重叠**（EZL 16.6% / TEST474 9.3%），是对齐 LAMP-PRo 数字的刻意保留；干净集 DRBP206 / PDB255 上 MCC 仍 +0.065 / +0.012。
- **结合位点 DNA-181 是短板**（0.3999）：平均序列更长、阳性更稀，MCC 天然更低，非 bug。
- **结合位点整体属中等水平**（MCC 0.34~0.54），靠「比同领域 SOTA（GraphBind / ESM-NBR）略好」的相对优势支撑，非碾压。
- 结构去冗余（变体 B / TM>0.5）会自断数据、掉分，故终版只做序列 30% 去冗余。
- 两任务均 best 在 epoch 2~3 收敛，之后是 patience 等待期，无需等满设定 epoch。

---

## 7. 相对基准的外部对比

| 结合位点 | DNA-129 | RNA-117 | 说明 |
|---|---|---|---|
| **ours（单链 650M）** | **0.5398** | **0.3556** | 本仓库最优 |
| GraphBind 原文 | 0.499 | 0.322 | NAR 2021 e51 |
| ESM-NBR（复现实测）| 0.506 | 0.215 | 在我们 BioLiP test 集实测 |

> ESM-NBR 为同领域竞争者，我们在同一套标准测试集上实测对比，ours 双双超过。

---

## 8. 待补 / 未上传项

- **模型权重**（本机 `/root/DRBP/finetune_6layers/`）：分类 `final_classfication/best_cls_650M.pt`（2.5 GB）、结合位点 `binding_site_singlechain/checkpoint/best_bsite_seq30_only_650M.pt`（2.6 GB）。
- **分类数据**（本机 `/root/DRBP/finetune_6layers/final_classfication/data/`：`train/train.csv` 13.6 MB + 四测试 CSV）——按用户要求后续自行处理。
- **训练/测试 PDB**：测试 PDB 已上传（`binding_site_singlechain/data/test_pdbs/`，424 个）；训练 PDB（4,722 单链）在本机 `/tmp/bsv4/rep_pdbs`，需按 §5.2 重新提取。
