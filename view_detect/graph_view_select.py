import os
import tqdm
import torch
from scene import Scene
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel

def object_views_select(labels, image_size, views, gaussians, pipeline, background, frame_num):

    visible_views = {}

    for ins_id in torch.unique(labels):
        for view_idx, view in enumerate(views):
            
            object_mask = labels == ins_id

            render_pkg= render(view, gaussians[object_mask], pipeline, background)
            means2D = render_pkg["info"]["means2d"]

            if means2D[:,0] > 0 and means2D[:,1] > 0 and means2D[:,0] < image_size[1] - 1 and means2D[:,1] < image_size[0] - 1:

                area = (means2D[:,0].max() - means2D[:,0].min()) * (means2D[:,1].max() - means2D[:,1].min())

                if ins_id not in visible_views:
                    visible_views[ins_id] = []
                visible_views[ins_id].append(view_idx, area)
            
            else:
                continue

    for ins_id, views in visible_views.items():
        
        selected_views = {}

        views.sort(key=lambda x: x[1], reverse=True)
        if len(views) < frame_num:
            selected_views[ins_id] = [view[0] for view in views]

        else:
            selected_views[ins_id] = [view[0] for view in views[:frame_num]]
            selected_views[ins_id] = list(set(selected_views[ins_id]))

    return selected_views

def object_pairs_views_select(labels, adj_matrix, image_size, views, gaussians, pipeline, background, frame_num):

    row_idx, col_idx = torch.triu_indices(adj_matrix.shape[0], adj_matrix.shape[1], offset=1, device=adj_matrix.device)
    adj_mask = adj_matrix[row_idx, col_idx]
    rel_pairs = torch.stack([row_idx[adj_mask], col_idx[adj_mask]], dim=1)  # [M,2]

    visible_views = {}
    for idx_i, idx_j in rel_pairs:

        for view_idx, view in enumerate(views):

            object_i_mask = labels == idx_i
            indices_i = torch.nonzero(object_i_mask, as_tuple=True)[0]
            object_j_mask = labels == idx_j
            indices_j = torch.nonzero(object_j_mask, as_tuple=True)[0]
            pair_mask = object_i_mask | object_j_mask
            ins_indices = torch.nonzero(pair_mask, as_tuple=True)[0]

            render_pkg= render(view, gaussians[pair_mask], pipeline, background)
            means2D = render_pkg["info"]["means2d"]

            # 所有物体相关的高斯球都被投影到该视角
            if means2D[:,0] > 0 and means2D[:,1] > 0 and means2D[:,0] < image_size[1] - 1 and means2D[:,1] < image_size[0] - 1:
        
                map_i = torch.insin(ins_indices, indices_i)
                map_j = torch.insin(ins_indices, indices_j)
                
                area = ((means2D[map_i][:,0].max() - means2D[map_i][:,0].min()) * (means2D[map_i][:,1].max() - means2D[map_i][:,1].min())
                        + (means2D[map_j][:,0].max() - means2D[map_j][:,0].min()) * (means2D[map_j][:,1].max() - means2D[map_j][:,1].min())) / 2

                if ins_id not in visible_views:
                    visible_views[ins_id] = []
                visible_views[ins_id].append(view_idx, area)
            
            else:
                continue

    for ins_id, views in visible_views.items():

        if len(views) == 0:
            assert False, "No visible views found for the instance pair."
        
        selected_views = {}
        views.sort(key=lambda x: x[1], reverse=True)
        
        if len(views) < frame_num:
            selected_views[ins_id] = [view[0] for view in views]

        else:
            selected_views[ins_id] = [view[0] for view in views[:frame_num]]
            selected_views[ins_id] = list(set(selected_views[ins_id]))
            
def process_views(args, dataset : ModelParams, opt : OptimizationParams, iteration : int, pipeline : PipelineParams, instance_labels : torch.Tensor, feature_level : int, view_type : str, adjactive_matrix : torch.Tensor):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, include_feature=True)
        checkpoint = os.path.join(args.model_path, f'chkpnt{iteration}_langfeat_{feature_level}_mask_single.pth')
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore_language_features(model_params, opt)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if view_type == 'caption':
            selected_views_dict = object_views_select(instance_labels, args.image_size, scene.getTrainCameras(), gaussians, pipeline, background, args.frame_num)
        
        elif view_type == 'relation':
            selected_views_dict = object_pairs_views_select(instance_labels, adjactive_matrix, args.image_size, scene.getTrainCameras(), gaussians, pipeline, background, args.frame_num)

        else:
            raise ValueError("Invalid view type. Choose either 'caption' or 'relation'.")

    return selected_views_dict