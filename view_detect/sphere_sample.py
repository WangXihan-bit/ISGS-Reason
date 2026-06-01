import torch
import math

def compute_bbox_sphere(bmin, bmax, scale=1.0):
    """Return the center and scaled enclosing-sphere radius of a 3D bbox."""

    center = 0.5 * (bmin + bmax)
    corners = torch.stack(torch.meshgrid(
        *[torch.tensor([bmin[i], bmax[i]], device=bmin.device) for i in range(3)]
    ), dim=-1).reshape(-1,3)
    radius = torch.norm(corners - center, dim=1).max()
    radius *= scale
    return center, radius

def quarter_sphere_equal(xyz, bbox, n_azim=3, n_elev=3, device="cuda"):
    """Sample camera centers on a local quarter sphere around the target bbox.

    The local frame is built from the scene center to the target center. Samples
    are biased toward the side facing the scene center, which is usually more
    consistent with indoor training-camera distributions than a full sphere.
    """
   
    bmin, bmax = bbox[:3], bbox[3:]
    target_center, radius = compute_bbox_sphere(bmin, bmax, scale=1.5)
    scene_center = torch.cat([xyz[:, :2].mean(dim=0), torch.zeros(1, device=xyz.device)])

    # Define the local view frame from a coarse scene center to the target.
    view_dir = target_center - scene_center
    view_dir = view_dir / (view_dir.norm() + 1e-6)

    # Avoid a degenerate right vector when the target direction is parallel to up.
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    if torch.allclose(torch.abs(torch.dot(view_dir, up)), torch.tensor(1.0, device=device)):
        up = torch.tensor([0.0, 1.0, 0.0], device=device)
    right = torch.cross(view_dir, up, dim=0)
    right = right / (right.norm() + 1e-6)
    local_up = torch.cross(right, view_dir, dim=0)
    local_up = local_up / (local_up.norm() + 1e-6)

    # Cover a limited azimuth and elevation band instead of sampling all sides.
    azim = torch.linspace(math.pi*1.2, math.pi*1.8, n_azim, device=device)  
    elev = torch.linspace(0.1, math.pi*0.4, n_elev, device=device) 
    azim, elev = torch.meshgrid(azim, elev, indexing="ij")

    x = torch.cos(elev) * torch.cos(azim)
    y = torch.cos(elev) * torch.sin(azim)
    z = torch.sin(elev)

    dirs = x[..., None] * right + y[..., None] * view_dir + z[..., None] * local_up
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-6)

    cam_positions = target_center[None, None, :] + dirs * radius
    return cam_positions.reshape(-1, 3)  # (N, 3)
