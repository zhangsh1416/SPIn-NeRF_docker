#!/bin/bash
set -e  # 出现错误时退出脚本

# 确保 conda 命令可用（根据实际情况修改路径）
source ~/anaconda3/etc/profile.d/conda.sh

#############################################
# 1. 主环境：spinnerf（用于主流程）
#############################################
echo "激活主 conda 环境：spinnerf"
conda activate spinnerf

#############################################
# 2. 数据集准备及初始 NeRF 运行（主流程）
#############################################
factor=1
dataset=bear

echo "生成相机参数 (运行 imgs2poses.py)..."
python imgs2poses.py data/$dataset

echo "===== 运行初始 NeRF (生成深度/disparity) ====="
echo "清空旧的 LaMa_test_images 和 output/label 文件夹..."
rm -rf lama/LaMa_test_images/*
rm -rf lama/output/label/*

echo "开始运行 DS_NeRF..."
python DS_NeRF/run_nerf.py \
    --config DS_NeRF/configs/config.txt \
    --render_factor 1 \
    --prepare \
    --i_weight 1000000000 \
    --i_video 1000000000 \
    --i_feat 4000 \
    --N_iters 4001 \
    --expname $dataset \
    --datadir ./data/$dataset \
    --factor $factor \
    --N_gt 0

#############################################
# 3. 切换到 LaMa 部分：进入 lama 目录并切换 conda 环境为 lama
#############################################
echo "===== 切换到 lama 部分 ====="
pushd lama

echo "激活 LaMa conda 环境：lama"
conda activate lama

echo "设置环境变量 TORCH_HOME 和 PYTHONPATH 为当前目录..."
export TORCH_HOME=$(pwd)
export PYTHONPATH=$(pwd)

echo "----- 运行 LaMa进行深度 inpainting -----"
python bin/predict.py refine=True model.path=$(pwd)/big-lama indir=$(pwd)/LaMa_test_images outdir=$(pwd)/output

echo "将生成的 inpainted disparity 复制到数据集中..."
rm -rf ../data/$dataset/images_${factor}/depth
mkdir -p ../data/$dataset/images_${factor}/depth
cp ./output/label/*.png ../data/$dataset/images_${factor}/depth

echo "----- 生成 inpainted RGB 图像 -----"
echo "清空旧的 LaMa_test_images 和 output/label..."
rm -rf lama/LaMa_test_images/*
rm -rf lama/output/label/*

echo "将原图复制到 LaMa_test_images..."
cp ../data/$dataset/images_${factor}/*.png LaMa_test_images
mkdir -p LaMa_test_images/label
cp ../data/$dataset/images_${factor}/label/*.png LaMa_test_images/label

echo "再次运行 LaMa进行 RGB inpainting..."
python bin/predict.py refine=True model.path=$(pwd)/big-lama indir=$(pwd)/LaMa_test_images outdir=$(pwd)/output

echo "将 inpainted RGB 图像复制到数据集中..."
rm -rf ../data/$dataset/images_${factor}/lama_images
mkdir -p ../data/$dataset/images_${factor}/lama_images
cp ../data/$dataset/images_${factor}/*.png ../data/$dataset/images_${factor}/lama_images
cp ./output/label/*.png ../data/$dataset/images_${factor}/lama_images

echo "完成 LaMa 部分，退出 lama 目录并切换回主环境..."

# 退出 lama 环境
conda deactivate
popd  # 返回仓库根目录

#############################################
# 4. 返回主环境后运行多视角 inpainting（主流程）
#############################################
echo "重新激活主 conda 环境：spinnerf"
conda activate spinnerf

echo "===== 运行多视角 inpainting ====="
python DS_NeRF/run_nerf.py \
    --config DS_NeRF/configs/config.txt \
    --i_feat 200 \
    --lpips \
    --i_weight 1000000000000 \
    --i_video 1000 \
    --N_iters 10001 \
    --expname $dataset \
    --train_scene 0 1 2 3 4 5 6 8 9 10 11 12 13 14 16 17 18 19 20 21 22 24 25 26 27 28 29 30 32 33 34 35 36 37 38 40 41 42 43 44 45 46 48 49 50 51 52 53 54 56 57 58 59 60 61 62 64 65 66 67 68 69 70 72 73 74 75 76 77 78 80 81 82 83 84 85 86 88 89 90 91 92 93 94\
    --test_scene 7 15 23 31 39 47 55 63 71 79 87 95 \  
    --datadir ./data/$dataset \
    --N_gt 0 \
    --factor $factor
python DS_NeRF/run_nerf.py 
    --config DS_NeRF/configs/config.txt 
    --lpips 
    --i_feat 200 
    --i_testset 1000 
    --i_weight 1000000000000 
    --i_video 1000 
    --N_iters 10001 
    --expname bear 
    --train_scene 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 
    --test_scene 1 49 
    --datadir ./data/bear 
    --N_gt 0 
    --factor 2

echo "===== 所有步骤执行完毕 ====="