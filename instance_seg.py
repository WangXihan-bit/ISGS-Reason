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
import cv2
import json
import torch

import colorsys
import numpy as np
from tqdm import tqdm
import open3d as o3d
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from argparse import ArgumentParser

from scene import Scene
from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state
from vis.vis_image_info import visualize_seg_map
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

def extract_gaussian_labels(model_path, iteration, source_path, views, gaussians, pipeline, background, feature_level, train_test_exp):

    language_feature_save_path = os.path.join(model_path, f'chkpnt{iteration}_langfeat_{feature_level}.pth')

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
       
        render_pkg= render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp)
        _, seg_map, _, depth_image = view.get_image_info(args, language_feature_dir=f"{source_path}/language_features", feature_level=feature_level)

        if depth_image is None:
            print(f"Warning: Depth image for {view.name} not found, skipping this view.")
            continue

        activated = render_pkg["info"]["activated"]
        means2D = render_pkg["info"]["means2d"]
        depths = render_pkg["info"]["depths"]
        mask = activated[0] > 0

        gaussians.assign_cluster_indices_dep(args, seg_map.permute(1, 2, 0), depth_image, mask, means2D[0,mask], depths[0,mask])

    valid_mask = gaussians._cluster_indices > 0 
    gaussians._cluster_indices = gaussians._cluster_indices[valid_mask]
    gaussians.prune_points(~valid_mask)
   
    cluster_indices = gaussians._cluster_indices.detach().cpu().numpy()
    xyz_np = gaussians._xyz.detach().cpu().numpy()
    
    for cluster_idx in np.unique(cluster_indices):

        mask = cluster_indices == cluster_idx
        xyz_idx_np = xyz_np[mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_idx_np)
        cl, ind = pcd.remove_radius_outlier(nb_points=10, radius=0.2)
        
        if len(ind) < 5:
            noise_indices = np.arange(len(xyz_idx_np))
            continue
        else:
            noise_indices = np.setdiff1d(np.arange(len(xyz_idx_np)), ind)
        gaussians._cluster_indices[torch.tensor(np.where(mask)[0][noise_indices],device=args.device)] = 0 

    torch.save((gaussians.capture_instance_feature(), 0), language_feature_save_path)
    print("checkpoint saved to: ", language_feature_save_path)
            
def process_scene_instance_labels(dataset : ModelParams, opt : OptimizationParams, iteration : int, pipeline : PipelineParams, feature_level : int):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, include_feature=False)

        checkpoint = os.path.join(args.model_path, f'chkpnt{iteration}.pth')
        (model_params, _) = torch.load(checkpoint, weights_only=False)
        gaussians.restore_rgb(model_params, opt)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
      
        extract_gaussian_labels(args.model_path, iteration, dataset.source_path, scene.getTrainCameras(), gaussians, pipeline, background, feature_level, dataset.train_test_exp)

def read_ply(file_path):
    points = np.load(file_path)
    print(points.shape, 'points shape')
    return points

if __name__ == "__main__":
    # Set up command line argument parser
    setproctitle.setproctitle("wxh Instance_Segmentation")
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--info_path", type=str, default="scene0000_00")
    parser.add_argument('--device', type=str, default="cuda:0")
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    process_scene_instance_labels(model.extract(args), opt.extract(args), args.iteration, pipeline.extract(args), args.feature_level)