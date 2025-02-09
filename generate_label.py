import os
import glob
import numpy as np
from PIL import Image
import argparse

def process_image(image_path, output_path):
    """
    处理单个图片：
      - 判断是否为单通道图片；
      - 检查像素值是否仅为 0 和 255；
      - 将所有像素值为 255 修改为 1，并保存结果。
    """
    # 打开图片
    try:
        im = Image.open(image_path)
    except Exception as e:
        raise RuntimeError(f"无法打开图片 {image_path}: {e}")

    # 检查图片是否为单通道
    if len(im.getbands()) != 1:
        raise ValueError(f"图片 {image_path} 不是单通道图片，通道数为 {len(im.getbands())}")

    # 转换为 numpy 数组进行处理
    im_arr = np.array(im)
    
    # 检查图片中唯一的像素值
    unique_vals = np.unique(im_arr)
    # 允许的像素值集合，仅允许 0 和 255
    allowed = {0, 255}
    if not set(unique_vals).issubset(allowed):
        raise ValueError(f"图片 {image_path} 中的像素值 {unique_vals} 不符合要求，允许的像素值仅为 0 和 255")
    
    # 将所有 255 修改为 1
    im_arr[im_arr == 255] = 1


    # 根据修改后的数组创建新的图片（模式与原图一致）
    new_im = Image.fromarray(im_arr)
    try:
        new_im.save(output_path)
    except Exception as e:
        raise RuntimeError(f"无法保存图片 {output_path}: {e}")

def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="处理单通道 png 图片，将所有像素值255改为1，并另存到同级目录下的 label 文件夹")
    parser.add_argument("input_folder", help="包含 png 图片的输入文件夹路径")
    args = parser.parse_args()
    
    input_folder = args.input_folder
    if not os.path.isdir(input_folder):
        parser.error(f"错误: 文件夹 {input_folder} 不存在")
    
    # 计算输入文件夹的同级目录下的 label 文件夹路径
    parent_dir = os.path.dirname(os.path.abspath(input_folder))
    output_folder = os.path.join(parent_dir, "label")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")
    else:
        print(f"输出文件夹已存在: {output_folder}")
    
    # 查找输入文件夹中所有 png 图片
    png_files = glob.glob(os.path.join(input_folder, "*.png"))
    if not png_files:
        parser.error("未在输入文件夹中找到 png 图片")
    
    # 处理每张图片
    for file in png_files:
        filename = os.path.basename(file)
        output_file = os.path.join(output_folder, filename)
        try:
            process_image(file, output_file)
            print(f"已处理 {filename}")
        except Exception as e:
            print(f"处理图片 {filename} 时出错: {e}")
            exit(1)

if __name__ == '__main__':
    main()