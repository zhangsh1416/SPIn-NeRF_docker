import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import read_write_model as read_model

def load_colmap_data(realdir):
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print('Cameras', len(camdata))  # 修复：应该是camdata的长度，不是cam

    h, w, f = cam.height, cam.width, cam.params[0]
    hwf = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    print('Images #', len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3D_binary(points3dfile)
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    return poses, pts3d, perm

def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            # 修复：移除过多的调试输出
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind-1] = 1
        vis_arr.append(cams)
    
    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print('Points', pts_arr.shape, 'Visibility', vis_arr.shape)
    
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr==1]
    print('Depth stats', valid_z.min(), valid_z.max(), valid_z.mean())
    
    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis==1]
        
        # 🔧 修复1: 正确的缩进
        if len(zs) == 0:
            print(f"[WARNING] No valid 3D points for image {i}, using default depths")
            close_depth, inf_depth = 0.1, 10.0
        else:
            # 🔧 修复2: 移除重复的百分位计算，使用保守值
            close_depth, inf_depth = np.percentile(zs, 0.5), np.percentile(zs, 99.5)
        
        # 🔧 修复3: 移除这行重复的极端百分位计算
        # close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    
    save_arr = np.array(save_arr)
    print(f"Generated poses_bounds with shape: {save_arr.shape}")  # 添加验证输出
    
    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)

def main(basedir, realdir):
    poses, pts3d, perm = load_colmap_data(realdir)
    save_poses(basedir, poses, pts3d, perm)

if __name__=='__main__':
    # 🔧 修复4: 修正路径，basedir应该指向redbull目录
    basedir = "data/data_WACV_spinnerf/redbull" 
    realdir = "data/data_WACV_spinnerf/redbull"
    main(basedir, realdir)