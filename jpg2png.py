import os
import argparse
from PIL import Image
import glob

def convert_jpg_to_png(file_path):
    """
    将单个 jpg 文件转换为 png 文件，并删除原 jpg 文件
    """
    try:
        # 打开 jpg 图片
        with Image.open(file_path) as im:
            # 构造新文件名：将扩展名由 .jpg 改为 .png
            base, _ = os.path.splitext(file_path)
            new_file = base + ".png"
            # 保存为 png 格式
            im.save(new_file, "PNG")
            print(f"转换完成: {file_path} -> {new_file}")
        # 删除原来的 jpg 文件，实现覆盖效果
        os.remove(file_path)
        print(f"已删除原文件: {file_path}")
    except Exception as e:
        print(f"处理 {file_path} 时出错: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="将目标文件夹内的jpg文件转换成png文件，覆盖原来图片"
    )
    parser.add_argument("target_folder", help="目标文件夹路径")
    args = parser.parse_args()

    # 检查目标文件夹是否存在
    if not os.path.isdir(args.target_folder):
        parser.error(f"目标文件夹 {args.target_folder} 不存在")

    # 查找所有 jpg 文件（同时支持小写和大写扩展名）
    jpg_files = glob.glob(os.path.join(args.target_folder, "*.jpg"))
    jpg_files += glob.glob(os.path.join(args.target_folder, "*.JPG"))

    if not jpg_files:
        print("目标文件夹中没有找到 jpg 文件")
        return

    # 遍历所有 jpg 文件进行转换
    for file_path in jpg_files:
        convert_jpg_to_png(file_path)

if __name__ == '__main__':
    main()