#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import time
import torch
import numpy as np
import open3d as o3d
import torch.nn.functional as F

from tqdm import tqdm

from argparse import ArgumentParser
from sklearn.neighbors import KDTree
from plyfile import PlyData, PlyElement
from scene import Scene
from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.ground_seg import get_ground_instance
from utils_image.clip_embed import OpenCLIPNetwork, OpenCLIPNetworkConfig
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

def extract_gaussian_features(model_path, iteration, source_path, views, gaussians, pipeline, background, feature_level):

    language_feature_save_path = os.path.join(model_path, f'chkpnt{iteration}_langfeat_{feature_level}_semantic.pth')
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        render_pkg= render(view, gaussians, pipeline, background)
        gt_language_feature, _, gt_mask, _= view.get_image_info(args, language_feature_dir=f"{source_path}/language_features", feature_level=feature_level)
        activated = render_pkg["info"]["activated"]
        significance = render_pkg["info"]["significance"]
        means2D = render_pkg["info"]["means2d"]
        mask = activated[0] > 0
        
        gaussians.accumulate_gaussian_feature_per_view(gt_language_feature.permute(1, 2, 0), gt_mask.permute(1, 2, 0), mask, significance[0,mask], means2D[0,mask])

    gaussians.finalize_gaussian_features()

    classes = {1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair", 6: "sofa", 7: "desk", 8: "door",
    9: "window", 10: "bookshelf", 11: "picture", 12: "counter", 14: "table", 16: "curtain", 24: "refrigerator", 
    28: "shower curtain", 29: "box", 33: "toilet", 34: "sink", 36: "bathtub"}

    texts = [f"{label}" for label in classes.values()]
    label_map = {idx: label for idx, label in enumerate(classes.keys())}
  
    clip_model = OpenCLIPNetwork(args, OpenCLIPNetworkConfig)
    texts_embed = clip_model.encode_texts(texts)
   
    valid_mask = gaussians._cluster_indices != 0 
   
    gs_feat = F.normalize(gaussians._language_feature, dim=1) 
    texts_embed = F.normalize(texts_embed, dim=1)
    similarity = gs_feat.float() @ texts_embed.T.float()
    max_sim, _ = similarity.max(dim=1)

    p_clip = torch.softmax(similarity / 1, dim=1)  

    # （1）entropy-based confidence
    H_i = -torch.sum(p_clip[valid_mask] * torch.log(p_clip[valid_mask] + 1e-8), dim=1)   
    w_H = 1.0 - H_i / torch.log(torch.tensor(p_clip.shape[1], device=p_clip.device, dtype=p_clip.dtype))

    # (2) intra-instance consistency
    num_instances = int(gaussians._cluster_indices.max().item()) + 1
    index_1d = gaussians._cluster_indices[valid_mask].squeeze()

    summed_feat = torch.zeros(num_instances, gs_feat.shape[1], device=gs_feat.device)
    count_feat = torch.zeros(num_instances, 1, device=gs_feat.device)
    conf_mask = max_sim[valid_mask] > 0.2
    valid_feats = gs_feat[valid_mask][conf_mask]
    valid_indices = index_1d[conf_mask]

    summed_feat.index_add_(0, valid_indices, valid_feats)
    count_feat.index_add_(0, valid_indices, torch.ones_like(valid_indices, dtype=torch.float).unsqueeze(1))

    # prototype feature
    mean_feat = summed_feat / (count_feat + 1e-8)
    mean_feat = F.normalize(mean_feat, dim=1)

    # cosine deviation
    feat_shift = 1 - (gs_feat[valid_mask] * mean_feat[index_1d]).sum(dim=1)
    sigma_f = feat_shift.mean().detach()
    w_F = torch.exp(-feat_shift**2 / (2 * sigma_f**2 + 1e-8))

    # (3) CAAR strategy
    H = H_i.detach()
    D = feat_shift.detach()
    rho = torch.corrcoef(torch.stack([H, D]))[0,1]
    alpha = torch.abs(rho) 
    w_i = alpha * w_F + (1 - alpha) * w_H
   
    # (4) distribution refinement
    q_m = torch.zeros(num_instances, p_clip.shape[1], device=p_clip.device)
    weight_sum = torch.zeros(num_instances, 1, device=p_clip.device)
    q_m.index_add_(0, index_1d, p_clip[valid_mask] * w_i.unsqueeze(1))
    weight_sum.index_add_(0, index_1d, w_i.unsqueeze(1))
    q_m = q_m / (weight_sum + 1e-8)

    # feature refinement
    gs_feat_distribution = p_clip.clone()
    gs_feat_distribution[valid_mask] = w_i.unsqueeze(1) * p_clip[valid_mask] + (1 - w_i.unsqueeze(1)) * q_m[index_1d]
    gs_feat_updated = gs_feat.clone()
    gs_feat_updated[valid_mask] = F.normalize(
        w_i.unsqueeze(1) * gs_feat[valid_mask] +
        (1 - w_i.unsqueeze(1)) * mean_feat[index_1d],
        dim=1
    )
    gaussians._language_feature = gs_feat_updated

    gs_semantic_label = torch.zeros(similarity.shape[0], dtype=torch.long, device=similarity.device)
    mask_confident = max_sim >= 0.1
    label_list = torch.tensor(list(label_map.values()), device=gs_feat_distribution.device)
    gs_semantic_label[mask_confident] = label_list[torch.argmax(gs_feat_distribution, dim=1)[mask_confident]] 

    torch.save((gaussians.capture_language_feature(), 0), language_feature_save_path)

def process_scene_language_features(dataset : ModelParams, opt : OptimizationParams, iteration : int, pipeline : PipelineParams, feature_level : int):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, include_feature=False)

        checkpoint = os.path.join(args.model_path, f'chkpnt{iteration}_langfeat_{feature_level}.pth')
        (model_params, _) = torch.load(checkpoint, weights_only=False)
        gaussians.restore_instance_features(model_params, opt)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        print("begin extracting language features...")
        extract_gaussian_features(args.model_path, iteration, dataset.source_path, scene.getTrainCameras(), gaussians, pipeline, background, feature_level)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gt_file", type=str, default="scene0000_00/scene0000_00_vh_clean_2.labels.ply")
    parser.add_argument("--info_path", type=str, default="scene0000_00/")
    parser.add_argument('--device', type=str, default="cuda:0")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    process_scene_language_features(model.extract(args), opt.extract(args), args.iteration, pipeline.extract(args), args.feature_level)

