import os
from platform import processor
import re
import cv2
import time
import torch
import torch_scatter
import setproctitle

import numpy as np
import open3d as o3d

from PIL import Image
from argparse import ArgumentParser
from gaussian_renderer import render
from gaussian_renderer import GaussianModel
from utils.graphics_utils import focal2fov
from utils.ground_seg import get_ground_instance
from query_view_select import render_view, render_set
from utils.camera_utils import cameraList_from_camInfos
from GroundingDINO.groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from sphere_sample import quarter_sphere_equal
from large_model_prompt import parse_relations, query_images, query_masks
from utils_image.clip_embed import OpenCLIPNetwork, OpenCLIPNetworkConfig
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def process_target_gs_ratio(gaussians, texts, ratio_thresh=0.7):

    clip_model = OpenCLIPNetwork(args, OpenCLIPNetworkConfig)

    texts_embed = clip_model.encode_texts(texts)  # [N_texts, D]
    similarity = gaussians._language_feature.float() @ texts_embed.float().T  # [N_gaussians, N_texts]
    
    sim_dist = similarity - similarity.min(dim=0, keepdim=True)[0]
    sim_dist = sim_dist / (sim_dist.max(dim=0, keepdim=True)[0] + 1e-6) 

    results = {}
    all_gs_valid_idx_list = []
    ground_id, _ = get_ground_instance(gaussians._xyz, gaussians._cluster_indices)

    for t_idx, text in enumerate(texts):

        gs_valid_idx = torch.tensor([], dtype=torch.long, device=gaussians._xyz.device)
        
        match_mask = sim_dist[:, t_idx] > ratio_thresh
        gaussian_idx = match_mask.nonzero(as_tuple=True)[0]
        instance_ids, counts = torch.unique(gaussians._cluster_indices[gaussian_idx], return_counts=True)

        valid_mask = (instance_ids != 0) & (instance_ids != ground_id)
        instance_ids = instance_ids[valid_mask]
        counts = counts[valid_mask]

        gs_valid_idx = instance_ids[counts > 50]

        all_gs_valid_idx_list.append(gs_valid_idx)

        results[text] = {
            "gs_valid_idx": gs_valid_idx   
        }

    all_gs_valid_idx = torch.unique(torch.cat(all_gs_valid_idx_list)) if all_gs_valid_idx_list else torch.tensor([], dtype=torch.long)

    return results, all_gs_valid_idx
    
def distance_matrix_center(gaussians, sub_idx, obj_idx):

    sub_centers = torch.stack([gaussians._xyz[gaussians._cluster_indices == idx].mean(dim=0) for idx in sub_idx])  # [Ns, 3]

    obj_centers = torch.stack([gaussians._xyz[gaussians._cluster_indices == idx].mean(dim=0) for idx in obj_idx])  # [No, 3]

    dist_matrix = torch.cdist(sub_centers, obj_centers)  # [Ns, No]

    return dist_matrix

def align_ptc_to_camera(point_cloud, align_matrix):
        """
        Reorients pointcloud to align with RGBD+pose camera data using provided realignment matrix.
        alignment matrix @ [x, y, z, 1]^T = [x', y', z', 1]^T
        inputs:
            point_cloud: N X 3
            align_matrix: 4 X 4
        returns:
            point_cloud: N X 3
        """

        point_cloud = torch.cat(
            [point_cloud, torch.ones_like(point_cloud[..., :1])], dim=-1
        )
        return torch.matmul(point_cloud, align_matrix.T)[..., :3]

def generate_bbox_refine(
    gaussians, region_dict, sigma=2.0, visualize=True
):
    
    xyz = gaussians._xyz
    cluster_idx = gaussians._cluster_indices

    xyz = gaussians._xyz
    bbox_dict = {}
    bbox_list, seg_list = [], []

    if visualize:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
        pc.paint_uniform_color([0.6, 0.6, 0.6])

    for text, info in region_dict.items():
        target_ids = info["gs_valid_idx"]
        if target_ids.numel() == 0:
            continue

        mask = torch.isin(cluster_idx, target_ids)
        xyz_target = xyz[mask]
        cluster_target = cluster_idx[mask]
        if xyz_target.numel() == 0:
            continue

        centroid_sum = torch_scatter.scatter_add(xyz_target, cluster_target, dim=0)
        counts = torch_scatter.scatter_add(
            torch.ones_like(cluster_target, dtype=torch.float32), cluster_target, dim=0
        )[:, None]
        centroids = centroid_sum / counts.clamp(min=1)

        centroids_per_point = centroids[cluster_target]
        dists = torch.norm(xyz_target - centroids_per_point, dim=1)
        mean_dists = torch_scatter.scatter_mean(dists, cluster_target, dim=0)
        std_dists = torch_scatter.scatter_std(dists, cluster_target, dim=0)
        thres = mean_dists + sigma * std_dists
        thres_per_point = thres[cluster_target]

        inlier_mask = dists < thres_per_point
        xyz_inliers = xyz_target[inlier_mask]
        cluster_inliers = cluster_target[inlier_mask]

        min_vals = torch_scatter.scatter_min(xyz_inliers, cluster_inliers, dim=0)[0]
        max_vals = torch_scatter.scatter_max(xyz_inliers, cluster_inliers, dim=0)[0]
        bbox_tensor = torch.cat([min_vals, max_vals], dim=1)

        bboxes = [bbox_tensor[idx].unsqueeze(0) for idx in target_ids if idx < bbox_tensor.shape[0]]
        if len(bboxes) > 0:
            bbox_dict[text] = torch.cat(bboxes, dim=0)
            bbox_dict[f"{text}_id"] = target_ids[target_ids < bbox_tensor.shape[0]]

        if visualize:
            for i, idx in enumerate(target_ids):
                if idx >= bbox_tensor.shape[0]:
                    continue
                bbox = o3d.geometry.AxisAlignedBoundingBox(
                    min_vals[idx].detach().cpu().numpy().astype(np.float64),
                    max_vals[idx].detach().cpu().numpy().astype(np.float64)
                )
                color = np.random.rand(3)
                bbox.color = color
                bbox_list.append(bbox)

                seg_pc = o3d.geometry.PointCloud()
                mask_cluster = (cluster_inliers == idx)
                seg_pc.points = o3d.utility.Vector3dVector(
                    xyz_inliers[mask_cluster].detach().cpu().numpy()
                )
                seg_pc.paint_uniform_color(color)
                seg_list.append(seg_pc)

    if visualize:
        o3d.visualization.draw_geometries([pc, *bbox_list, *seg_list])

    return bbox_dict

def bbox_iou_3d_batch(boxes1, boxes2):
   
    min1, max1 = boxes1[:, None, :3], boxes1[:, None, 3:]  # [N,1,3]
    min2, max2 = boxes2[None, :, :3], boxes2[None, :, 3:]  # [1,N,3]

    inter_min = torch.max(min1, min2)
    inter_max = torch.min(max1, max2)
    inter_size = (inter_max - inter_min).clamp(min=0)
    inter_vol = inter_size.prod(dim=-1)  # [N,N]

    vol1 = (max1 - min1).prod(dim=-1).squeeze(1)  # [N]
    vol2 = (max2 - min2).prod(dim=-1).squeeze(0)  # [N]

    union_vol = vol1[:, None] + vol2[None, :] - inter_vol
    iou = inter_vol / (union_vol + 1e-6)
    return iou

def merge_bboxes_iou(bboxes, sub_id_sets, iou_thresh=0.3):
    
    N = bboxes.shape[0]
    visited = torch.zeros(N, dtype=torch.bool, device=bboxes.device)
    merged_list = []
    merged_sub_id_sets = []

    iou_mat = bbox_iou_3d_batch(bboxes, bboxes)

    for i in range(N):
        if visited[i]:
            continue

        group_idx = (iou_mat[i] > iou_thresh).nonzero(as_tuple=False).squeeze(1)
        visited[group_idx] = True

        group_boxes = bboxes[group_idx]
        merged_min = group_boxes[:, :3].min(dim=0).values
        merged_max = group_boxes[:, 3:].max(dim=0).values
        merged_list.append(torch.cat([merged_min, merged_max], dim=0).unsqueeze(0))

        merged_ids = set()
        for idx in group_idx.tolist():
            merged_ids |= sub_id_sets[idx]
        merged_sub_id_sets.append(merged_ids)

    merged_bboxes = torch.cat(merged_list, dim=0)

    return merged_bboxes, merged_sub_id_sets

def build_region_items(sub_bbox_dict, obj_bbox_dict):
    region_items = []

    # sub
    for k, v in sub_bbox_dict.items():
        if k.endswith("_id"):
            continue
        region_items.append({
            "name": k,
            "bbox": v,
            "id": sub_bbox_dict.get(f"{k}_id", None),
            "is_sub": True
        })

    # obj
    for k, v in obj_bbox_dict.items():
        if k.endswith("_id"):
            continue
        region_items.append({
            "name": k,
            "bbox": v,
            "id": obj_bbox_dict.get(f"{k}_id", None),
            "is_sub": False
        })

    region_items = sorted(region_items, key=lambda x: x["bbox"].shape[0])
    return region_items

def init_sub_id_sets(root_item):
    root_bboxes = root_item["bbox"]
    root_ids = root_item["id"]

    if root_item["is_sub"] and root_ids is not None:
        return [{int(root_ids[j].item())} for j in range(root_bboxes.shape[0])]
    else:
        return [set() for _ in range(root_bboxes.shape[0])]

def root2leaf(gaussians, sub_bbox_dict, obj_bbox_dict, top_k=3):

    region_items = build_region_items(sub_bbox_dict, obj_bbox_dict)

    root_item = region_items[0]
    merged_bboxes = root_item["bbox"].clone()
    merged_sub_id_sets = init_sub_id_sets(root_item)

    for i in range(1, len(region_items)):
        leaf_node = region_items[i]
        leaf_bboxes = leaf_node["bbox"]
        leaf_ids = leaf_node["id"]

        center_leaf = (leaf_bboxes[:, :3] + leaf_bboxes[:, 3:]) / 2
        center_merged = (merged_bboxes[:, :3] + merged_bboxes[:, 3:]) / 2

        dist_mat = torch.cdist(center_merged, center_leaf, p=2)
        closest_idx = torch.argmin(dist_mat, dim=0)

        selected_merged = merged_bboxes[closest_idx]            # [N_leaf, 6]
        merged_min = torch.min(selected_merged[:, :3], leaf_bboxes[:, :3])
        merged_max = torch.max(selected_merged[:, 3:], leaf_bboxes[:, 3:])
        merged_bboxes = torch.cat([merged_min, merged_max], dim=1)  # [N_leaf, 6]

        new_sub_id_sets = []
        for k in range(leaf_bboxes.shape[0]):
            parent_idx = int(closest_idx[k].item())
            cur_set = set(merged_sub_id_sets[parent_idx])  
            if leaf_node["is_sub"] and leaf_ids is not None:
                cur_set.add(int(leaf_ids[k].item()))

            new_sub_id_sets.append(cur_set)

        merged_bboxes, merged_sub_id_sets = merge_bboxes_iou(merged_bboxes, new_sub_id_sets, iou_thresh=0.5)
  
    sizes = merged_bboxes[:, 3:] - merged_bboxes[:, :3]
    volumes = sizes.prod(dim=1)  

    xyz = gaussians._xyz
    scene_min = xyz.min(dim=0).values
    scene_max = xyz.max(dim=0).values
    scene_size = scene_max - scene_min
    scene_vol = scene_size.prod()
    vol_thresh = scene_vol / 4

    valid_mask = volumes < vol_thresh
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

    # 再排序，选前 top_k
    sorted_idx = torch.argsort(volumes[valid_idx])
    topk_idx = valid_idx[sorted_idx[:top_k]]
    topk_bboxes = merged_bboxes[topk_idx]
    topk_sub_ids = [merged_sub_id_sets[int(i.item())] for i in topk_idx]

    return topk_bboxes, topk_sub_ids

def visualize_masks_with_ids(image_pil, seg_masks, contain_tol=0):
    
    image = np.array(image_pil).copy()
    N = len(seg_masks)

    rng = np.random.default_rng(2024)
    colors = rng.integers(0, 255, (N, 3), dtype=np.uint8)

    overlay = image.copy()

    bboxes = []
    for i in range(N):
        mask = seg_masks[i]
        ys, xs = np.where(mask == 1)
        if len(xs) == 0:
            bboxes.append(None)
            continue

        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        bboxes.append({
            "orig_id": i,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "area": area
        })

    remove_ids = set()
    for i in range(N):
        if bboxes[i] is None:
            continue
        bi = bboxes[i]

        for j in range(N):
            if i == j or bboxes[j] is None:
                continue
            bj = bboxes[j]

            contained = (
                bi["x1"] <= bj["x1"] + contain_tol and
                bi["y1"] <= bj["y1"] + contain_tol and
                bi["x2"] >= bj["x2"] - contain_tol and
                bi["y2"] >= bj["y2"] - contain_tol
            )

            if contained and bi["area"] > bj["area"]:
                remove_ids.add(j)

    kept_boxes = []
    kept_indices = []
    for i in range(N):
        if i in remove_ids or bboxes[i] is None:
            continue
        kept_boxes.append(bboxes[i])
        kept_indices.append(i)

    id_map = {}
    for new_id, box in enumerate(kept_boxes, start=1):
        id_map[new_id] = box["orig_id"]

    for new_id, box in enumerate(kept_boxes, start=1):
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        color = colors[box["orig_id"]].tolist()

        cv2.putText(
            overlay,
            str(new_id),
            (x1 + 5, y1 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

    return Image.fromarray(overlay), kept_indices, id_map

def merge_bbox(prompt):

    for i, p in enumerate([prompt]):
        print(f"\nPrompt {i}: {p}")
        relations = parse_relations(args, p)
        print("Parsed relations:", relations)

        texts_sub = [rel['subject'] for rel in relations["triples"]]
        uni_sub = [t for t in set(texts_sub) if t.strip() != '']
        assert len(uni_sub) != 0
        sub_dict, sub_idx = process_target_gs_ratio(gaussians, uni_sub, ratio_thresh=0.7)
        
        texts_obj = [rel.get('object', '') for rel in relations["triples"]]
        uni_obj = [t for t in set(texts_obj) if isinstance(t, str) and t.strip() != '']

        if len(uni_obj) > 0:
            obj_dict, obj_idx = process_target_gs_ratio(gaussians, uni_obj, ratio_thresh=0.7)
        else:
            obj_dict = {}
            obj_idx = torch.empty(0, dtype=gaussians._cluster_indices.dtype,
                          device=gaussians._cluster_indices.device)

        refine_obj_bbox = generate_bbox_refine(gaussians, obj_dict, visualize=False)
        refine_sub_bbox = generate_bbox_refine(gaussians, sub_dict, visualize=False)
        
        merged_bbox, merged_sub_ids = root2leaf(gaussians, refine_sub_bbox, refine_obj_bbox)

    return merged_bbox, relations, merged_sub_ids, uni_sub

def generate_views(merged_bbox):
    anchor_center = (merged_bbox[:, :3] + merged_bbox[:, 3:]) / 2

    image_size = [968, 1296]
    focal_length = torch.tensor([1169.621094, 1167.105103]).to(args.device) 
    
    FovY = focal2fov(focal_length[1], image_size[1])
    FovX = focal2fov(focal_length[0], image_size[0])

    for i, bbox in enumerate(merged_bbox):
        print(f"Generating view for bbox {i}")
        cam_pose = quarter_sphere_equal(gaussians._xyz, bbox)
        scene_center = gaussians._xyz.mean(dim=0)

        if cam_pose.dim() == 1:
            cam_pose = cam_pose.unsqueeze(0)

        view_centers = torch.cat(
            [scene_center.unsqueeze(0), cam_pose],
            dim=0
        )
     
        view_tags = ["scene"] + [f"sphere_{j}" for j in range(cam_pose.shape[0])]

        cam_infos, c2w = render_view(
            centers=view_centers,
            anchor_bboxes=anchor_center,
            FovX=FovX,
            FovY=FovY,
            save_dir=args.save_dir,
            image_size=image_size,
            bbox_id=i,
            view_tags=view_tags,
        )
     
    views = cameraList_from_camInfos(cam_infos, 1.0, args, True)

    return views

def refine_target(relations, merged_bbox, gaussians, sub_ids, uni_sub, views, ini_thresh=0.5):
    
    best_score, best_image_id = query_images(relations["triples"], os.path.join(args.model_path, 'query_view_renders'))
    
    match = re.search(r'(\d+)', best_image_id)
    if match:
        best_idx = int(match.group(1))
    else:
        raise ValueError(f"Invalid image_id format: {best_image_id}")
    
    if best_score == 0:

        if len(merged_bbox) == 1:
            print("path 1")
            target_sub_id = sub_ids[0]
            target_ids = torch.tensor(
                list(target_sub_id),
                device=gaussians._cluster_indices.device,
                dtype=gaussians._cluster_indices.dtype
            )

            target_mask = torch.isin(gaussians._cluster_indices, target_ids)
            print(target_mask.sum())
            xyz_np = gaussians._xyz.detach().cpu().numpy()
            mask_xyz = xyz_np[target_mask.detach().cpu().numpy()]  # (N,3)
            # 计算 3D AABB
            bbox_min = mask_xyz.min(axis=0)
            bbox_max = mask_xyz.max(axis=0)
          
        else:
            print("Warning: No good view found, using feature similarity to select target bbox.")
            return None, None
    else:

        print("path 2: " + best_image_id)
        target_image = Image.open(os.path.join(args.model_path, 'query_view_renders', best_image_id))

        sam3_model = build_sam3_image_model(
                                            bpe_path=args.bpe_path,
                                            checkpoint_path=args.sam_ckpt,
                                            device=args.device
                                            ).to(args.device)
  
        prompt = uni_sub[0]
        thresh = ini_thresh

        while True:
            processor = Sam3Processor(sam3_model, confidence_threshold=thresh)
            inference_state = processor.set_image(target_image)
            processor.reset_all_prompts(inference_state)

            output = processor.set_text_prompt(state=inference_state, prompt=prompt)
            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

            if len(masks) > 0:
                break
            thresh *= 0.8
  
        best_view = views[best_idx]
        rendering = render(best_view, gaussians, pipeline, background, use_trained_exp=model.extract(args).train_test_exp, 
                           separate_sh=args.separate_sh, include_depth=True)
        rendered_depth = rendering["render"][3]
        depths = rendering["info"]["depths"].squeeze(0)
        activated = rendering["info"]["activated"]
        means2D = rendering["info"]["means2d"].squeeze(0)

        if len(masks) == 1:
            print("2-1")     
            mask_2d = masks[0].squeeze()

        else:
            print("2-2")
            seg_masks = masks.squeeze(1).cpu().numpy()  # [N, H, W]
            result_img, kept_indices, id_map = visualize_masks_with_ids(target_image, seg_masks)

            image_save_path = os.path.join(args.model_path, 'mask_vis', 'mask_vis.png')

            if not os.path.exists(os.path.join(args.model_path, 'mask_vis')):
                os.makedirs(os.path.join(args.model_path, 'mask_vis'))

            result_img.save(image_save_path)

            best_mask_id = query_masks(relations["triples"], image_save_path, list(id_map.keys()))
            best_mask_id = int(best_mask_id)

            if best_mask_id not in id_map:
                raise ValueError(f"best_mask_id={best_mask_id} is invalid")

            orig_mask_idx = id_map[best_mask_id] 
            mask_2d = masks[orig_mask_idx].squeeze()

        save_dir_mask2d = os.path.join(args.model_path, 'mask_vis')
        os.makedirs(save_dir_mask2d, exist_ok=True)
        img_np = np.array(target_image)   # RGB, [H, W, 3]
        mask_np = mask_2d.detach().cpu().numpy()
        mask_np = mask_np > 0

        overlay = img_np.copy()
        overlay[mask_np] = [255, 0, 0]   

        vis = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)
        cv2.imwrite(
            os.path.join(save_dir_mask2d, 'selected_mask_overlay.png'),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        )

        H, W = mask_2d.shape[:2]
        obj_mask = torch.flip(mask_2d.to(args.device).bool(), dims=[1])  

        visible_idx = (activated[0] > 0).nonzero(as_tuple=True)[0]
        
        xy = means2D[visible_idx].round().long()
        proj_depths = depths[visible_idx]
     
        batch_y = torch.clamp(xy[:, 1], min=0, max=H - 1)
        batch_x = torch.clamp(xy[:, 0], min=0, max=W - 1)
        gt_depth = rendered_depth[batch_y.long(), batch_x.long()].float() 
        valid_gs_depth = torch.isclose(gt_depth, proj_depths, rtol=0.03)

        valid_mask = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H) & valid_gs_depth
      
        xy = xy[valid_mask]
        visible_idx = visible_idx[valid_mask]

        inside_mask = obj_mask[xy[:, 1], xy[:, 0]]
        gaussian_idx_in_mask = visible_idx[inside_mask]

        xyz_np = gaussians._xyz.detach().cpu().numpy()
        mask_xyz = xyz_np[gaussian_idx_in_mask.detach().cpu().numpy()]  # (N,3)
   
        bbox_min = mask_xyz.min(axis=0)
        bbox_max = mask_xyz.max(axis=0)

    return bbox_min, bbox_max
        
if __name__ == '__main__':
    setproctitle.setproctitle('wxh query')
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--separate_sh", type=bool, default=False)
    parser.add_argument('--sam_model_type', default="vit_h")
    parser.add_argument('--sam_ckpt', default="sam3/chechpoints/sam3.pt")
    parser.add_argument("--bpe_path", type=str, default="sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--save_dir", type=str, default="/output/scene0000_00")
    parser.add_argument("--info_path", type=str, default="scene0000_00")
    parser.add_argument("--ply", type=str, default="scene0000_00/points3d.ply")
    parser.add_argument("--depth_path", type=str, default="scene0000_00/depth/")
    parser.add_argument('--spatial_model_name', type=str, default="vila-siglip-llama-3b")
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = get_combined_args(parser)

    setproctitle.setproctitle("wxh test")

    with torch.no_grad():
        gaussians = GaussianModel(model.extract(args).sh_degree)

        checkpoint = os.path.join(args.model_path, f'chkpnt{args.iteration}_langfeat_{args.feature_level}_semantic.pth')
        (model_params, _) = torch.load(checkpoint, weights_only=False)
        gaussians.restore_language_features(model_params, opt.extract(args))

    # test example
    prompt_example = "Facing the bike, the table on the left"
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    infer_start = time.perf_counter()
    merged_bbox, relations, sub_idx, uni_sub = merge_bbox(prompt_example)

    views = generate_views(merged_bbox)
    
    bg_color = [1,1,1] if model.extract(args).white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        gaussians_raw = GaussianModel(model.extract(args).sh_degree)

        checkpoint = os.path.join(args.model_path, f'chkpnt{args.iteration}.pth')
        (model_params, _) = torch.load(checkpoint, weights_only=False)
        gaussians_raw.restore_rgb(model_params, opt.extract(args))
  
    render_set(args.save_dir, "query_view_renders", views, gaussians_raw, pipeline, background, model.extract(args).train_test_exp, args.separate_sh)
    save_path = os.path.join(args.save_dir, 'refined_target.ply')
    bbox_min, bbox_max = refine_target(relations, merged_bbox, gaussians, sub_idx, uni_sub, views)

        

           

