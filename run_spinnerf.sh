#!/bin/bash
set -e  # 出现错误时退出脚本

# 日志文件路径（可根据需要修改）
LOG_FILE="run_log.txt"

# 在日志文件中添加分隔头，标识一次新实验开始
echo -e "\n\n### New Experiment: $(date '+%Y-%m-%d %H:%M:%S') ###" >> "${LOG_FILE}"

# 数据集列表
datasets=(7 3 book)

for dataset in "${datasets[@]}"; do
  # 记录开始时间
  start_time=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$start_time] START dataset=${dataset}" | tee -a "${LOG_FILE}"

  # 执行命令，并将输出追加到日志文件
  python DS_NeRF/run_nerf.py \
    --config DS_NeRF/configs/config.txt \
    --i_feat 200 \
    --lpips \
    --i_weight 10000 \
    --i_video 1000 \
    --i_testset 1000 \
    --N_iters 10001 \
    --expname spinnerf_${dataset}_4_with_test \
    --train_scene 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 \
    --test_scene 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 \
    --datadir ./data/spinnerf-dataset/${dataset} \
    --N_gt 0 \
    --factor 4 \
  2>&1 | tee -a "${LOG_FILE}"

  # 记录结束时间
  end_time=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$end_time] END   dataset=${dataset}" | tee -a "${LOG_FILE}"
done