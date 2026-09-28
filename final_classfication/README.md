# 分类（DNA/RNA 结合蛋白多标签）— 650M 最终交付

> 最后更新：2026-09-23 · 自包含目录，含最优 checkpoint、训练脚本、推理脚本

预测一个蛋白是 **DBP（DNA 结合）/ RBP（RNA 结合）/ DRBP（双结合）/ non** 的多标签分类。
序列用 ESM-2 编码，三个标签头（DBP / RBP / DRBP，`head_scheme=non` 时第三头为 non、DRBP 由
DBP∧RBP 派生）。本次相对 150M 版**只换 backbone：ESM-2 150M → 650M，架构与配方完全不变**。

---

## 最终结果（四个标准测试集）

**最优模型 `best_cls_650M.pt`**（epoch 3，searched 阈值），对比 150M 同架构同配方：

| 测试集 | 指标 | 150M | 650M | Δ |
|---|---|---|---|---|
| **DRBP206** | dbp_auc | 0.7608 | 0.7113 | −0.049 |
| (206) | rbp_auc | 0.8628 | 0.8348 | −0.028 |
| | drbp_auc | 0.8378 | 0.7690 | −0.069 |
| | **MCC** | 0.4897 | **0.5551** | **+0.065** |
| | DRBP 捕获 | 44/103 | **53/103** | +9 |
| **PDB255** | dbp_auc | 0.8809 | **0.9007** | +0.020 |
| (255) | rbp_auc | 0.8721 | **0.8765** | +0.004 |
| | **MCC** | 0.5456 | **0.5576** | +0.012 |
| **TEST474** | dbp_auc | 0.9829 | 0.9619 | −0.021 |
| (474) | rbp_auc | 0.9429 | 0.9272 | −0.016 |
| | drbp_auc | 0.8439 | 0.7634 | −0.081 |
| | **MCC** | 0.7823 | **0.7844** | +0.002 |
| **EZL** | dbp_auc | 0.9636 | **0.9748** | +0.011 |
| (4003) | rbp_auc | 0.9723 | **0.9846** | +0.012 |
| | **MCC** | 0.7425 | **0.7724** | +0.030 |

> 评测口径：best-threshold MCC（阈值在验证集上搜索），与 150M 版、train_drbp 最终评估
> 完全一致。括号内为各测试集蛋白数。

### 关键结论

- **MCC 四个测试集全面胜出**（硬判定，最贴近实际应用）。最大亮点是 **DRBP206 +0.065**
  （0.490→0.555）——该集专门测 DRBP（DNA+RNA 双结合），是历史最难啃的短板，DRBP 捕获从
  44 涨到 53（+9 个真双结合蛋白）；最大集 **EZL +0.030**。
- **AUC（排序）混合**：DBP/RBP 主头在 PDB255、EZL 略胜，在 DRBP206、TEST474 略负
  （−0.02～−0.08）。
- **验证集全面领先、测试集 AUC 未完全迁移**：验证集上 650M 每个 epoch 全线压过 150M
  （best 点 d/r/b = 0.9805/0.9708/0.8743 vs 150M 0.9776/0.9676/0.8622），但独立测试集
  AUC 变成有高有低。说明 650M 更强的表征让硬判定更准，排序能力却没有稳定超出 150M。
- 一个可留意点：DRBP206 的 MCC 大涨以 drbp_auc −0.069 为代价——650M 的 DRBP 头排序更差，
  阈值搜索用"多喊 DRBP"换回更高召回（44→53），是阈值 tradeoff 而非纯粹更准。

---

## 模型

```
序列分支: ESM-2 (esm2_t33_650M_UR50D, hidden 1280, 33 层, 解冻最后 6 层)
主干:     SharedTrunk (多尺度 Conv1d + MHSA + 门控残差, d_model 256, dropout 0.3)
pooling:  LabelQueryPooling (每标签一个可学习 query 各自 attend 序列)
标签间:   MaskedCrossLabelAttention (DBP↔RBP 互看, DRBP 看所有, 无门控)
头:       三个独立两层 MLP → DBP / RBP / DRBP (head_scheme=non 时第三头为 non)
损失:     pos_weight BCE + DRBP 直接监督 + 层级一致性 + 共现边际
```

- **可训练参数 119,888,772（≈1.2 亿）**：解冻 ESM 后 6 层 + 分类头。前 27 层（≈5.3 亿
  参数）冻结，来自 ESM-2 在数千万蛋白序列上的预训练，不参与下游拟合——这是回应
  "650M 会不会过拟合"的关键：有效自由度是 1.2 亿可训练参数，不是 6.5 亿。
- 相对 150M 版唯一变化：`esm_model` 150M→650M、`esm_embed_dim` 640→1280（`train_drbp.py`
  已改为从实际 ESM 读 `esm.config.hidden_size`，两版本自动适配）。

---

## 训练配置（复现）

```bash
cd /root/DRBP/finetune_6layers/final_classfication
/root/.conda/envs/drbp/bin/python train_drbp.py --n_gpus 4 \
  --train_csv train_final.csv --mode esm2 --tag finetune6_650M \
  --epochs 30 --patience 12 --batch_size 2 --accum 2 --lr 1e-4 \
  --select_metric composite --select_ema 0.5 --no_test_every_best \
  --head_scheme non --drbp_upsample 5.0 --unfreeze_esm_layers 6 --no_gate \
  --esm_model /root/DRBP/models/esm2_t33_650M_UR50D \
  --save best_cls_650M.pt
```

> `batch_size=2`（150M 是 8）：650M 在 P100 12G 下只塞得下 bs=2，每 epoch 约 47 分钟。
> best epoch 3 后 patience=12 的等待期可自行收紧（150M/650M 均 best 在 epoch 3）。

## 推理 / 评估（四个测试集）

```bash
cd /root/DRBP/finetune_6layers/final_classfication
/root/.conda/envs/drbp/bin/python eval_650m_test.py --ckpt best_cls_650M.pt
```

输出 DRBP206 / PDB255 / TEST474 / EZL 四个测试集的 `dbp_auc / rbp_auc / drbp_auc / MCC / DRBP 捕获`。

---

## 文件结构

```
final_classfication/
├── best_cls_650M.pt        # 最优 checkpoint (epoch 3, 2.5G, 完整 650M 权重)
├── train_drbp.py           # DDP 训练脚本 (4 卡 fork 启动)
├── eval_650m_test.py       # 四测试集推理/评估脚本
├── model_drbp.py           # DRBPNetNew + 损失
├── model_baseline.py       # Config / SharedTrunk / 门控残差
├── metrics.py              # 指标计算 + 阈值搜索
├── struct_adapter.py       # 结构适配层 (esm2 模式未用, import 依赖保留)
└── data -> /root/DRBP/new/data   # 数据软链 (train_final.csv + 四测试集 CSV)
```

环境必须用 `/root/.conda/envs/drbp`（torch cu118）；HuggingFace 调用已带 `local_files_only=True`。

---

## 已知边界

- **AUC 未稳定超 150M**：如果论文/汇报以 AUROC 为主指标，650M 的优势不成立，需说明；
  若以 MCC（硬判定）为主指标，650M 四集全胜、DRBP 短板改善，值得采用。
- 训练在 best epoch 3 已收敛，之后是 patience 等待期；本次评估即用 ep3 权重，无需等满 30 epoch。
