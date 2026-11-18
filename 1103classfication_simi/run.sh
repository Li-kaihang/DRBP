#!/bin/bash
# DRBP-Net 两 query 分类 (head_scheme='two', 无 NDRBP 头) —— 最优架构 × 「simi」口径。
#
# 与最优 checkpoint best_twoquery.pt 的训练命令完全一致（classfication_twoquery/run.sh），
# 唯一区别：--train_csv train_withsimi.csv（22181 条，去 100% 重复、保留部分同源），
# 用于补全「固定架构下数据口径阶梯」的中间档：
#   train_final(22892, 含一模一样) → train_withsimi(22181) → train_clean30(19795, 严格)。
set -e
cd "$(dirname "$0")"

PY=/root/.conda/envs/drbp/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PY train_drbp.py \
  --n_gpus 4 \
  --train_csv train_withsimi.csv \
  --mode esm2 \
  --tag no_non_650M_withsimi \
  --epochs 30 --patience 12 \
  --batch_size 2 --accum 2 \
  --lr 1e-4 \
  --select_metric composite --select_ema 0.5 \
  --no_test_every_best \
  --head_scheme two \
  --drbp_upsample 5.0 \
  --unfreeze_esm_layers 6 \
  --no_gate \
  --esm_model /root/DRBP/models/esm2_t33_650M_UR50D
