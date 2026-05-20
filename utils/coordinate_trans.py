import os
import cv2
import numpy as np
import torch

# 3d -> 2d projection
def transform_pt_depth_scannet_torch(points, depth_intrinsic, depth, pose, device):
    """
    :param points: N x 3 format
    :param depth: H x W format
    :param intrinsic: 3x3 format
    :return p: N x 2 format
    """

    vis_thres = 0.1
    depth_shift = 1000.0
    
    # 内参：intrinsic.txt
    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx = depth_intrinsic[0,3]
    by = depth_intrinsic[1,3]
    
    points_world = torch.cat([points, torch.ones((points.shape[0], 1), dtype=torch.float64).to(device)], dim=-1).to(torch.float64)
    world_to_camera = torch.inverse(pose)
    
    p = torch.matmul(world_to_camera, points_world.T)  # [Xb, Yb, Zb, 1]: 4, n
    p[0] = ((p[0] - bx) * fx) / p[2] + cx 
    p[1] = ((p[1] - by) * fy) / p[2] + cy
    
    all_idx = torch.arange(0, len(points)).to(device)  # to save the corresponding point idx later as the prompt ID
    # out-of-image check
    idx = torch.unique(torch.cat([torch.where(p[0]<=0)[0], torch.where(p[1]<=0)[0], \
                                    torch.where(p[0]>=depth.shape[1]-1)[0], \
                                    torch.where(p[1]>=depth.shape[0]-1)[0]], dim=0), dim=0)
    keep_idx = all_idx[torch.isin(all_idx, idx, invert=True)]
    p = p[:, keep_idx]

    if p.shape[1] == 0:
        return p, keep_idx  # no 3D prompt is visible in this frame
        
    # Simply round the final coordinates into pixel value
    pi = torch.round(p).to(torch.int64)
    # Check occlusion
    est_depth = p[2]
    trans_depth = depth[pi[1], pi[0]] / depth_shift
    idx_keep = torch.where(torch.abs(est_depth - trans_depth) <= vis_thres)[0]
    
    p = p.T[idx_keep, :2]
    keep_idx = keep_idx[idx_keep]
    
    return p, keep_idx

# 2d -> 3d backprojection
def backproject_2d_to_3d_scannet(points, depth, depth_intrinsic, pose, device):
   
    u = points[:, 0].long()  # shape: (N,)
    v = points[:, 1].long()  # shape: (N,)

    depth_shift = 1000.0  
    depth_value = depth.to(torch.float64)[v, u]
    z = depth_value / depth_shift

    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]

    x = ((u - cx) * z) / fx 
    y = ((v - cy) * z) / fy
    ones = torch.ones_like(z)
    camera_points = torch.stack([x, y, z, ones], dim=1).to(torch.float64).to(device)  # (N, 4)

    world_points = (pose @ camera_points.T).T[:, :3]  # (N, 3)
    
    return world_points

# 2d-3d mapping
def compute_mapping(points, data_path, scene_name, frame_id):  
    """
    :param points: N x 3 format
    :param depth: H x W format
    :param intrinsic: 3x3 format
    :return: mapping, N x 3 format, (H,W,mask)
    """
    vis_thres = 0.1
    depth_shift = 1000.0

    mapping = np.zeros((3, points.shape[0]), dtype=int)
    
    # Load the intrinsic matrix
    depth_intrinsic = np.loadtxt(os.path.join(data_path, 'intrinsics.txt'))
    
    # Load the depth image, and camera pose
    depth = cv2.imread(os.path.join(data_path, scene_name, 'depth', frame_id + '.png'), -1) # read 16bit grayscale 
    pose = np.loadtxt(os.path.join(data_path, scene_name, 'pose', frame_id + '.txt' ))

    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx = depth_intrinsic[0,3]
    by = depth_intrinsic[1,3]
    
    points_world = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
    world_to_camera = np.linalg.inv(pose)
    p = np.matmul(world_to_camera, points_world.T)  # [Xb, Yb, Zb, 1]: 4, n
    p[0] = ((p[0] - bx) * fx) / p[2] + cx 
    p[1] = ((p[1] - by) * fy) / p[2] + cy
    
    # out-of-image check
    mask = (p[0] > 0) * (p[1] > 0) \
                    * (p[0] < depth.shape[1]-1) \
                    * (p[1] < depth.shape[0]-1)

    pi = np.round(p).astype(int) # simply round the projected coordinates
    
    # directly keep the pixel whose depth!=0
    depth_mask = depth[pi[1][mask], pi[0][mask]] != 0
    mask[mask == True] = depth_mask
    
    # occlusion check:
    trans_depth = depth[pi[1][mask], pi[0][mask]] / depth_shift
    est_depth = p[2][mask]
    occlusion_mask = np.abs(est_depth - trans_depth) <= vis_thres
    mask[mask == True] = occlusion_mask

    mapping[0][mask] = p[1][mask]
    mapping[1][mask] = p[0][mask]
    mapping[2][mask] = 1

    return mapping.T