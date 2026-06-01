import os
import math
import torch
import numpy as np

from typing import NamedTuple
from camera_param import look_at_view_transform

from os import makedirs
from gaussian_renderer import render
import torchvision

from tqdm import tqdm

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    cx: np.array
    cy: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    is_test: bool

def get_c2w(R, T):
   
    R_c2w = R.T
    t_c2w = -R.T @ T
    c2w = torch.eye(4, dtype=R.dtype, device=R.device)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = t_c2w
    return c2w

def render_view(
    centers,
    anchor_bboxes,
    FovX,
    FovY,
    save_dir=None,
    image_size=[640, 480],
    bbox_id=0,
    view_tags=None,
    global_uid=0,
):
    """
    Generate camera infos from both scene-center view and sphere-sampled views.

    Args:
        centers: Tensor, [K, 3] or [3], camera centers / view positions.
        anchor_bboxes: Tensor, [N, ...], target anchor bboxes.
        FovX, FovY: camera FoV.
        save_dir: directory to save rendered images.
        image_size: [W, H].
        bbox_id: current bbox index, used for filename.
        view_tags: list of view names, e.g., ["scene", "sphere_0", ...].

    Returns:
        cam_infos: list of CameraInfo.
        c2w: list of camera-to-world matrices.
    """

    cam_infos = []
    c2w = []

    os.makedirs(save_dir, exist_ok=True)

    # 保证 centers 是 [K, 3]
    if centers.dim() == 1:
        centers = centers.unsqueeze(0)

    if view_tags is None:
        view_tags = [f"view_{i}" for i in range(centers.shape[0])]

    anchor_bbox_3d = anchor_bboxes

    R, T, cx, cy = setup_camera(
        anchor_bbox_3d=anchor_bbox_3d.unsqueeze(0),
        center=centers,
        camera_distance_factor=0,
        camera_lift=0,
    )

    # 如果只生成一个视角，统一转成 batch 形式，方便后续处理
    if R.dim() == 2:
        R = R.unsqueeze(0)
        T = T.unsqueeze(0)

    for j in range(R.shape[0]):
        R_v = R[j]
        T_v = T[j]

        mat = torch.eye(4, dtype=R_v.dtype, device=R_v.device)
        mat[:3, :3] = R_v
        mat[:3, 3] = T_v

        c2w.append(mat)

        w2c = np.linalg.inv(mat.detach().cpu().numpy())

        # R is stored transposed due to 'glm' in CUDA code
        R_cam = np.transpose(w2c[:3, :3])
        T_cam = w2c[:3, 3]

        view_name = view_tags[j]

        image_name = f"render_bbox{bbox_id}_{view_name}.jpg"
        image_path = os.path.join(save_dir, image_name)

        cam_infos.append(
            CameraInfo(
                uid=global_uid + j,
                R=R_cam,
                T=T_cam,
                FovY=FovY,
                FovX=FovX,
                cx=cx,
                cy=cy,
                image_path=image_path,
                image_name=image_name,
                width=image_size[0],
                height=image_size[1],
                is_test=True,
            )
        )

    return cam_infos, c2w

def render_set(model_path, name, views, gaussians, pipeline, background, train_test_exp, separate_sh):

    render_path = os.path.join(model_path, name)
    makedirs(render_path, exist_ok=True)

    for fname in os.listdir(render_path):
        fpath = os.path.join(render_path, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
    
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def setup_camera(
    anchor_bbox_3d,
    center,
    camera_distance_factor=0,
    camera_lift=0,
):
    """
    Set up the camera for rendering the point cloud.

    Args:
        point_cloud (Pointclouds): The point cloud to render.
        anchor_bbox_3d (torch.Tensor): The 3D bounding box of the anchor.
        center (np.ndarray): The center of the point cloud.
        image_size (int): The size of the output image.
        camera_distance_factor (float): The factor to adjust camera distance.
        camera_lift (float): The lift to apply to the camera.
        device (str): The device to use for computation.
        calibrate (bool): Whether to calibrate the camera.

    Returns:
        PerspectiveCameras: The set up camera.
    """

    center = torch.tensor(center, dtype=torch.float32)
    center[2] += camera_lift
    camera_position = center + camera_distance_factor * (center - anchor_bbox_3d)
   
    R, T = look_at_view_transform(
        dist=1,
        elev=0,
        azim=0,
        eye=camera_position,
        at=anchor_bbox_3d,
        up=((0, 0, -1),),
    )

    principal_point = torch.tensor([320, 240]).to(
        center.device
    )  # Initial principal point, shape (1, 2)

    cx = principal_point[0]
    cy = principal_point[1]

    return R, T, cx, cy
