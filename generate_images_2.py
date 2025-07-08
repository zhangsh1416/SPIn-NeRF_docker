import os
import glob
import argparse
from PIL import Image

def downsample_image(image_path, output_path):
    """
    对单张图片进行两倍降采样：
      - 打开图片；
      - 计算新尺寸（宽、高各减半）；
      - 利用高质量的 Lanczos 重采样方法降采样；
      - 保存降采样后的图片到 output_path。
    """
    try:
        im = Image.open(image_path)
    except Exception as e:
        raise RuntimeError(f"无法打开图片 {image_path}: {e}")

    # 获取原图尺寸
    width, height = im.size
    new_size = (width // 2, height // 2)
    
    if new_size[0] < 1 or new_size[1] < 1:
        raise ValueError(f"图片 {image_path} 太小，无法进行降采样")

    # 使用 Lanczos 滤镜进行降采样
    downsampled_im = im.resize(new_size, Image.LANCZOS)
    try:
        downsampled_im.save(output_path)
    except Exception as e:
        raise RuntimeError(f"无法保存图片 {output_path}: {e}")

def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(
        description="对输入文件夹中的图片进行两倍降采样，并另存到同级目录下的 images_2 文件夹中"
    )
    parser.add_argument("input_folder", help="包含待处理图片的输入文件夹路径")
    args = parser.parse_args()

    input_folder = args.input_folder
    if not os.path.isdir(input_folder):
        parser.error(f"输入文件夹 {input_folder} 不存在")

    # 计算输入文件夹同级目录下的 images_2 文件夹路径
    parent_dir = os.path.dirname(os.path.abspath(input_folder))
    output_folder = os.path.join(parent_dir, "labels_2")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")
    else:
        print(f"输出文件夹已存在: {output_folder}")

    # 支持的图片文件后缀（可根据需要扩展）
    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))

    if not image_files:
        parser.error("未在输入文件夹中找到图片")

    # 对每张图片进行降采样处理
    for image_path in image_files:
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, filename)
        try:
            downsample_image(image_path, output_path)
            print(f"已处理 {filename}")
        except Exception as e:
            print(f"处理图片 {filename} 时出错: {e}")
            exit(1)

if __name__ == '__main__':
    main()