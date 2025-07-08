#!/bin/bash
set -e  # 出现错误时退出脚本

# 日志文件路径（可根据需要修改）
LOG_FILE="run_log.txt"

# 在日志文件中添加分隔头，标识一次新实验开始
echo -e "\n\n### New Experiment: $(date '+%Y-%m-%d %H:%M:%S') ###" >> "${LOG_FILE}"

#############################################
# 参数设置
#############################################
factor=1
dataset=bear
total_images=96

# 根据总图片数生成训练视图和测试视图参数
train_scene=""
test_scene=""
for (( i=0; i<total_images; i++ )); do
    if (( (i+1) % 8 == 0 )); then
         test_scene+="$i "
    else
         train_scene+="$i "
    fi
done

#############################################
# 记录开始时间及参数信息
#############################################
start_time=$(date '+%Y-%m-%d %H:%M:%S')
start_epoch=$(date +%s)

{
    echo "============================"
    echo "脚本开始时间: ${start_time}"
    echo "参数信息:"
    echo "  factor: ${factor}"
    echo "  dataset: ${dataset}"
    echo "  total_images: ${total_images}"
    echo "  train_scene: ${train_scene}"
    echo "  test_scene: ${test_scene}"
    echo "  Python 命令: "
    echo "    python DS_NeRF/run_nerf.py \\"
    echo "      --config DS_NeRF/configs/config.txt \\"
    echo "      --i_feat 200 \\"
    echo "      --lpips \\"
    echo "      --i_weight 10000 \\"
    echo "      --i_video 1000 \\"
    echo "      --N_iters 10001 \\"
    echo "      --i_testset 1000 \\"
    echo "      --expname ${dataset}_2_bash_RTXA6000_screen_weight \\"
    echo "      --train_scene ${train_scene}\\"
    echo "      --test_scene ${test_scene}\\"
    echo "      --datadir ./data/${dataset} \\"
    echo "      --N_gt 0 \\"
    echo "      --factor ${factor}"
    echo "----------------------------"
} | tee -a "${LOG_FILE}"

#############################################
# 运行 NeRF 脚本
#############################################
python DS_NeRF/run_nerf.py \
    --config DS_NeRF/configs/config.txt \
    --i_feat 200 \
    --lpips \
    --i_weight 10000 \
    --i_video 1000 \
    --N_iters 10001 \
    --i_testset 1000 \
    --expname ${dataset}_1_bash_RTXA6000_screen_weight \
    --train_scene ${train_scene} \
    --test_scene ${test_scene} \
    --datadir ./data/${dataset} \
    --N_gt 0 \
    --factor ${factor} 2>&1 | tee -a "${LOG_FILE}"

#############################################
# 记录结束时间及运行时长
#############################################
end_time=$(date '+%Y-%m-%d %H:%M:%S')
end_epoch=$(date +%s)
duration=$(( end_epoch - start_epoch ))

{
    echo "----------------------------"
    echo "脚本结束时间: ${end_time}"
    echo "运行时长: ${duration} 秒"
    echo "============================"
} | tee -a "${LOG_FILE}"