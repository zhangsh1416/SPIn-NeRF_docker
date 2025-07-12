# 创建debug_load_colmap_depth.py
import numpy as np
import sys
import os
from pathlib import Path

sys.path.append('DS_NeRF')

def debug_load_colmap_depth():
    basedir = './data/data_WACV_spinnerf/bag'
    factor = 2
    bd_factor = 0.75
    prepare = True
    
    print(f"🔍 调试load_colmap_depth函数执行...")
    print(f"参数: basedir={basedir}, factor={factor}, bd_factor={bd_factor}, prepare={prepare}")
    
    try:
        # 1. 检查_load_data是否正常
        print(f"\n步骤1: 测试_load_data函数...")
        from load_llff import _load_data
        
        result = _load_data(basedir, factor=factor, prepare=prepare)
        if result is None:
            print("❌ _load_data返回None!")
            return False
        else:
            print(f"✅ _load_data成功，返回{len(result)}个值")
            _, bds_raw, imgs, _, _, _, _, _ = result
            total_images = imgs.shape[0]
            print(f"总图像数: {total_images}")
        
        # 2. 测试COLMAP数据读取
        print(f"\n步骤2: 测试COLMAP数据读取...")
        from colmapUtils.read_write_model import read_images_binary, read_points3d_binary
        
        images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
        points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')
        
        print(f"COLMAP图像数: {len(images)}")
        print(f"COLMAP 3D点数: {len(points)}")
        print(f"图像ID范围: {min(images.keys())} - {max(images.keys())}")
        
        # 3. 手动执行load_colmap_depth的核心逻辑
        print(f"\n步骤3: 手动执行深度计算逻辑...")
        
        from load_llff import get_poses
        poses = get_poses(images)
        print(f"poses形状: {poses.shape}")
        
        # 计算误差统计
        Errs = np.array([point3D.error for point3D in points.values()])
        Err_mean = np.mean(Errs)
        print(f"平均重投影误差: {Err_mean}")
        
        # 计算缩放因子
        bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
        sc = 1. if bd_factor is None else 1. / (bds_raw.min() * bd_factor)
        print(f"缩放因子: {sc}")
        
        # 4. 逐张图像检查深度计算
        print(f"\n步骤4: 检查前5张图像的深度计算...")
        
        data_list = []
        successful_images = 0
        
        for id_im in range(1, min(6, len(images) + 1)):  # 只检查前5张
            if id_im not in images:
                print(f"图像 {id_im}: 不存在于COLMAP中")
                continue
                
            print(f"\n图像 {id_im} ({images[id_im].name}):")
            
            depth_list = []
            coord_list = []
            weight_list = []
            
            total_2d_points = len(images[id_im].xys)
            valid_3d_refs = sum(1 for pid in images[id_im].point3D_ids if pid != -1)
            existing_3d_points = sum(1 for pid in images[id_im].point3D_ids 
                                   if pid != -1 and pid in points)
            
            print(f"  2D点数: {total_2d_points}")
            print(f"  有效3D引用: {valid_3d_refs}")
            print(f"  实际存在的3D点: {existing_3d_points}")
            
            for i in range(len(images[id_im].xys)):
                point2D = images[id_im].xys[i]
                id_3D = images[id_im].point3D_ids[i]
                
                if id_3D == -1:
                    continue
                if id_3D not in points:
                    print(f"  警告: 3D点 {id_3D} 不存在")
                    continue
                
                point3D = points[id_3D].xyz
                depth = (poses[id_im - 1, :3, 2].T @ (point3D - poses[id_im - 1, :3, 3])) * sc
                
                # 检查深度范围
                if depth < bds_raw[id_im - 1, 0] * sc or depth > bds_raw[id_im - 1, 1] * sc:
                    continue
                
                err = points[id_3D].error
                weight = 2 * np.exp(-(err / Err_mean) ** 2)
                
                depth_list.append(depth)
                coord_list.append(point2D / factor)
                weight_list.append(weight)
            
            print(f"  最终有效深度点: {len(depth_list)}")
            
            if len(depth_list) > 0:
                successful_images += 1
                data_list.append({
                    "depth": np.array(depth_list),
                    "coord": np.array(coord_list),
                    "weight": np.array(weight_list)
                })
                print(f"  ✅ 成功生成深度数据")
            else:
                print(f"  ❌ 没有生成有效深度数据")
                data_list.append({
                    "depth": np.array([]),
                    "coord": np.array([]).reshape(0, 2),
                    "weight": np.array([])
                })
        
        print(f"\n📊 测试结果:")
        print(f"成功处理的图像数: {successful_images}/5")
        print(f"生成的数据项数: {len(data_list)}")
        
        if successful_images == 0:
            print(f"❌ 关键问题: 没有任何图像生成有效深度数据!")
            print(f"可能原因:")
            print(f"1. 深度值超出边界范围")
            print(f"2. 所有3D点都被过滤掉")
            print(f"3. poses计算错误")
            
            # 检查边界值
            print(f"\n检查边界值:")
            print(f"bds_raw范围: {bds_raw.min()} - {bds_raw.max()}")
            print(f"缩放后边界: {bds_raw.min() * sc} - {bds_raw.max() * sc}")
        
        return successful_images > 0
        
    except Exception as e:
        print(f"❌ 调试过程失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_load_colmap_depth()