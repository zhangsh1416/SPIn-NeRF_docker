import cv2
import numpy as np
import os
import glob
import argparse

def process_images(input_folder, output_folder):
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历文件夹内所有png文件
    for file_path in glob.glob(os.path.join(input_folder, "*.png")):
        # 读取图片（以不做颜色转换的方式读取）
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"无法读取 {file_path}")
            continue

        # 检查是否为单通道图片（灰度图），若不是，则跳过或做适当处理
        if len(img.shape) != 2:
            print(f"图片 {file_path} 不是单通道图片，已跳过。")
            continue

        # 根据要求将像素值转换：像素值等于2，3，5的pixel修改为1，其它全部修改为0
        processed_img = np.where((img == 4), 1, 0).astype(np.uint8)



        # 构造输出文件名并保存图片
        base_name = os.path.basename(file_path)
        output_path = os.path.join(output_folder, base_name)
        cv2.imwrite(output_path, processed_img)
        print(f"处理并保存了 {output_path}")

    print("所有图片处理完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理文件夹中所有png图片：检查是否为单通道，将像素值转换并下采样")
    parser.add_argument("input_folder", help="输入图片所在文件夹路径")
    parser.add_argument("output_folder", help="输出图片保存的文件夹路径")
    args = parser.parse_args()

    process_images(args.input_folder, args.output_folder)