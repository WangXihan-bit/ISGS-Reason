import os
import re
import json
import numpy as np

def load_intrinsics(intrinsic_file):
    intr = np.loadtxt(intrinsic_file).reshape(4, 4)
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    return fx, fy, cx, cy

def load_poses(pose_path):

    def trailing_number(fname):
        name = os.path.splitext(os.path.basename(fname))[0]
        m = re.search(r'(\d+)$', name)
        if m: return int(m.group(1))
        nums = re.findall(r'\d+', name)
        return int(nums[-1]) if nums else -1
    
    poses = []
    for pose_file in sorted(os.listdir(pose_path), key=trailing_number):
        pose_pth = os.path.join(pose_path, pose_file)
        mat = np.loadtxt(pose_pth).reshape(4, 4)
        mat[:,1:3] *= -1
        poses.append(mat)
    return poses

def scannet_to_json(intrinsic_file, pose_file, out_json, w, h, step=20):
    fx, fy, cx, cy = load_intrinsics(intrinsic_file)
    poses = load_poses(pose_file)

    frames = []
    for i, pose in enumerate(poses):
        color_idx = i * step
        
        frame = {
            "file_path": f"color/{color_idx}",
            "transform_matrix": pose.tolist()
        }
        frames.append(frame)

    out = {
        "w": w,
        "h": h,
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "frames": frames
    }

    with open(out_json, "w") as f:
        json.dump(out, f, indent=4)

scannet_to_json(
    intrinsic_file="scene0000_00/intrinsic/intrinsic_color.txt",
    pose_file="scene0000_00/pose/",
    out_json="scene0000_00/transforms_train_raw.json",
    w=1296, h=968
)

scannet_to_json(
    intrinsic_file="scene0000_00/intrinsic/intrinsic_color.txt",
    pose_file="scene0000_00/pose/",
    out_json="scene0000_00/transforms_test_raw.json",
    w=1296, h=968
)
