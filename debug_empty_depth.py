import numpy as np
import sys
import os
from pathlib import Path

# 在模块级别添加路径和导入
sys.path.append('DS_NeRF')

def debug_empty_depth_gts():
    basedir = 'data/data_WACV_spinnerf/bag'
    
    print("🔍 调试depth_gts为空的问题...")
    
    # 1. 检查colmap_depth.npy
    depth_file = Path(basedir) / 'colmap_depth.npy'
    if depth_file.exists():
        try:
            depth_gts = np.load(depth_file, allow_pickle=True)
            print(f"colmap_depth.npy存在")
            print(f"类型: {type(depth_gts)}")
            print(f"长度: {len(depth_gts) if hasattr(depth_gts, '__len__') else 'N/A'}")
            if hasattr(depth_gts, '__len__') and len(depth_gts) > 0:
                print(f"第一个元素: {depth_gts[0]}")
                print(f"第一个元素类型: {type(depth_gts[0])}")
            else:
                print("❌ depth_gts为空!")
        except Exception as e:
            print(f"❌ 加载colmap_depth.npy失败: {e}")
    else:
        print("❌ colmap_depth.npy文件不存在")
    
    # 2. 检查COLMAP数据
    print(f"\n🔍 检查COLMAP数据...")
    sparse_dir = Path(basedir) / 'sparse' / '0'
    
    if not sparse_dir.exists():
        print("❌ sparse/0目录不存在!")
        return False
    
    # 检查文件存在性
    files = ['cameras.bin', 'images.bin', 'points3D.bin']
    for f in files:
        filepath = sparse_dir / f
        exists = filepath.exists()
        size = filepath.stat().st_size if exists else 0
        print(f"{f}: {'✅' if exists else '❌'} ({size} bytes)")
    
    try:
        from colmapUtils.read_write_model import read_images_binary, read_points3d_binary
        
        images = read_images_binary(sparse_dir / 'images.bin')
        points = read_points3d_binary(sparse_dir / 'points3D.bin')
        
        print(f"\nCOLMAP数据统计:")
        print(f"图像数: {len(images)}")
        print(f"3D点数: {len(points)}")
        
        # 检查图像-3D点关联
        valid_images = 0
        for i, (img_id, img_data) in enumerate(list(images.items())[:5]):
            valid_3d_points = sum(1 for pid in img_data.point3D_ids 
                                if pid != -1 and pid in points)
            print(f"图像 {img_id} ({img_data.name}): {len(img_data.point3D_ids)} 2D点, {valid_3d_points} 有效3D点")
            if valid_3d_points > 0:
                valid_images += 1
        
        print(f"有有效3D点的图像数: {valid_images}/5 (样本)")
        
        if valid_images == 0:
            print("❌ 关键问题: 图像与3D点没有有效关联!")
            print("这就是depth_gts为空的原因!")
        
        return valid_images > 0
        
    except Exception as e:
        print(f"❌ COLMAP数据读取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_empty_depth_gts()