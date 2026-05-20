import os
import json
import open3d as o3d
import numpy as np
from plyfile import PlyData, PlyElement

def load_axis_alignment(txt_path):
    with open(txt_path, "r") as f:
        first_line = f.readline().strip() 
    
    values_str = first_line.split("=")[1].strip()
    values = list(map(float, values_str.split()))
    
    axis_alignment = np.array(values).reshape(4, 4)
    return axis_alignment

def apply_axis_alignment(json_in, axis_align, json_out):

    with open(json_in, "r") as f:
        data = json.load(f)

    for frame in data["frames"]:
        tm = np.array(frame["transform_matrix"], dtype=np.float64)
       
        new_tm = axis_align @ tm   # 左乘
        new_tm = np.round(new_tm, 6)
        frame["transform_matrix"] = new_tm.tolist()

    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)

def align_ply(ply_in, axis_alignment, ply_out):

    plydata = PlyData.read(ply_in)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T   # (N,3)

    N = positions.shape[0]
    positions_h = np.hstack([positions, np.ones((N, 1))])  # (N,4)

    positions_aligned = (positions_h @ axis_alignment.T)[:, :3]

    has_color = all(c in vertices.data.dtype.names for c in ('red', 'green', 'blue'))
    if has_color:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T.astype(np.uint8)
        vertex_data = np.array(
            list(zip(
                positions_aligned[:, 0], positions_aligned[:, 1], positions_aligned[:, 2],
                colors[:, 0], colors[:, 1], colors[:, 2]
            )),
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        )
    else:
        vertex_data = np.array(
            list(zip(
                positions_aligned[:, 0], positions_aligned[:, 1], positions_aligned[:, 2]
            )),
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        )

    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el], text=True).write(ply_out)

def align_ply_o3d(ply_in, axis_alignment, ply_out):

    pcd = o3d.io.read_point_cloud(ply_in)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    N = points.shape[0]
    points_h = np.hstack([points, np.ones((N, 1))])  # (N, 4)

    points_aligned = (axis_alignment @ points_h.T).T[..., :3]

    pcd.points = o3d.utility.Vector3dVector(points_aligned)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 保存
    o3d.io.write_point_cloud(ply_out, pcd)

def render_pointcloud_image(points, colors, K, extrinsic, image_size):

    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    N = points.shape[0]
    points_h = np.hstack([points, np.ones((N, 1))])  # (N, 4)

    points_cam = extrinsic @ points_h.T
    points_cam[0] = ((points_cam[0]) * fx) / points_cam[2] + cx
    points_cam[1] = ((points_cam[1]) * fy) / points_cam[2] + cy

    H, W = image_size
    img = np.zeros((H, W, 3), dtype=np.uint8)

    for i in range(points_cam.shape[0]):
        u, v = int(points_cam[i, 0]), int(points_cam[i, 1])
        if 0 <= u < W and 0 <= v < H:
            img[v, u] = (colors[i] * 255).astype(np.uint8)

    return img

base_path = "scene0000_00"
axis_alignment = load_axis_alignment(os.path.join(base_path, 'scene0000_00.txt'))
apply_axis_alignment(os.path.join(base_path, 'transforms_train_raw.json'), axis_alignment, os.path.join(base_path, 'transforms_train.json'))
apply_axis_alignment(os.path.join(base_path, 'transforms_test_raw.json'), axis_alignment, os.path.join(base_path, 'transforms_test.json'))
align_ply(os.path.join(base_path, 'points3d_raw.ply'), axis_alignment, os.path.join(base_path, 'points3d.ply'))       
         

