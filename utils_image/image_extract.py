import os
import cv2
import numpy as np
import torch
from torchvision.utils import save_image

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def update_seg_images_from_seg_maps(seg_maps, image):
  
    seg_images = {}
    bbox_images = {}

    for mode in seg_maps.keys():
        seg_img_list = []
        bbox_img_list = []
        seg_map = seg_maps[mode]
        
        unique_indices = np.unique(seg_map)
    
        for idx in unique_indices:
       
            if idx == -1:
                continue  
            image_copy = image.copy()
            mask = (seg_map == idx)
   
            y_indices, x_indices = np.where(mask)  
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()

            bbox_img = image_copy[y_min:y_max+1, x_min:x_max+1]
            image_copy[~mask] = np.array([0, 0, 0], dtype=np.uint8) 
            seg_img = image_copy[y_min:y_max+1, x_min:x_max+1] 
            
            pad_bbox_img = cv2.resize(pad_img(bbox_img), (224, 224))
            pad_seg_img = cv2.resize(pad_img(seg_img), (224, 224))

            seg_img_list.append(pad_seg_img)
            bbox_img_list.append(pad_bbox_img)

        seg_imgs = np.stack(seg_img_list, axis=0)
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0, 3, 1, 2) / 255.0).to('cuda')
        seg_images[mode] = seg_imgs

        bbox_imgs = np.stack(bbox_img_list, axis=0)
        bbox_imgs = (torch.from_numpy(bbox_imgs.astype("float32")).permute(0, 3, 1, 2) / 255.0).to('cuda')
        bbox_images[mode] = bbox_imgs

    return seg_images, bbox_images

def remove_sparse_contour_pixels(seg_map, min_pixel_threshold=10):

    unique_indices = np.unique(seg_map)
    
    for idx in unique_indices:
        if idx == -1:  
            continue

        mask = (seg_map == idx).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for label in range(1, num_labels): 
            pixel_count = stats[label, cv2.CC_STAT_AREA]
            if pixel_count < min_pixel_threshold:
                seg_map[labels == label] = -1  # 删除稀疏像素点
    
    return seg_map

def compute_iou_matrix(generated_masks, seg_map, unique_mask_indices):
   
    num_seg_masks = len(unique_mask_indices)
    generated_masks = generated_masks.astype(np.bool_)

    iou_matrix = np.zeros((1, num_seg_masks))

    for i, mask_index in enumerate(unique_mask_indices):
        
        if mask_index == -1:
            continue

        seg_mask = (seg_map == mask_index)  # (H, W)
        
        intersection = np.sum(generated_masks & seg_mask) 
        union = np.sum(generated_masks | seg_mask) 

        iou_matrix[:, i] = intersection / (union + 1e-6)  

    return iou_matrix

def compute_iou_matrix_from_dict(generated_mask_dict, seg_map, unique_mask_indices):
   
    gen_obj_ids = list(generated_mask_dict.keys())
    gen_masks = np.stack([generated_mask_dict[obj_id] for obj_id in gen_obj_ids]) 
    gen_masks = gen_masks.astype(np.bool_)
    
    num_gen_masks = len(gen_obj_ids)
    num_seg_masks = len(unique_mask_indices)
    iou_matrix = np.zeros((num_gen_masks, num_seg_masks), dtype=np.float32)

    for j, seg_id in enumerate(unique_mask_indices):
        if seg_id == -1:  
            continue

        seg_mask = (seg_map == seg_id)  

        intersection = np.logical_and(gen_masks, seg_mask).sum(axis=(1, 2)) 
        union = np.logical_or(gen_masks, seg_mask).sum(axis=(1, 2))        

        iou_matrix[:, j] = intersection / (union + 1e-6)

    return iou_matrix, gen_obj_ids

def get_bbox_img(box, image):
    image = image.copy()
    x_min, y_min, x_max, y_max = map(int, box)

    seg_img = image[y_min:y_max, x_min:x_max]
    return seg_img

def relabel_seg_map(seg_map_np, ignore_val=-1):
 
    unique_ids = np.unique(seg_map_np)
    valid_ids = unique_ids[unique_ids != ignore_val]
    
    id_map = {old_id: new_id for new_id, old_id in enumerate(valid_ids)}
    new_seg_map = np.full_like(seg_map_np, ignore_val)

    for old_id, new_id in id_map.items():
        new_seg_map[seg_map_np == old_id] = new_id

    return new_seg_map, id_map

def sam_predictor(args, predictor_sam, seg_map, image, detections):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    visualized_image = image.copy()
    predictor_sam.set_image(image=image)
    mask_indices = []
    seg_masks = []

    unique_mask_indices_cache = {}
    for mode in ['default']:
        unique_mask_indices_cache[mode] = np.unique(seg_map[mode])

    for i, box in enumerate(detections.xyxy):
        predictor_sam.set_image(image=image)
        masks, scores, logits = predictor_sam.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        generated_mask = masks[index]
     
        seg_masks.append(generated_mask.astype(np.uint8))

        mode_mask_indices = {}
        for mode, unique_mask_indices in unique_mask_indices_cache.items():

            iou_matrix = compute_iou_matrix(generated_mask[None, :, :], seg_map[mode], unique_mask_indices)
            best_mask_index = unique_mask_indices[np.argmax(iou_matrix)]

            mode_mask_indices[mode] = best_mask_index
    
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(generated_mask.astype(np.uint8), kernel, iterations=1)
            updated_region = dilated_mask.astype(bool) 

            seg_map[mode][updated_region] = best_mask_index

        mask_indices.append(mode_mask_indices)

    for mode in ['default']:
        seg_map[mode] = remove_sparse_contour_pixels(seg_map[mode], min_pixel_threshold=20)
        seg_map[mode], _ = relabel_seg_map(seg_map[mode], ignore_val=-1)

    return mask_indices, seg_map, seg_masks, visualized_image

def iou_track(mask, track_masks):
    obj_ids = list(track_masks.keys())
    track_masks = np.stack([track_masks[obj_id] for obj_id in obj_ids], axis=0) # (num_masks, H, W)
    intersection = np.logical_and(track_masks, mask[None, :, :]).sum(axis=(1, 2))
    union = np.logical_or(track_masks, mask[None, :, :]).sum(axis=(1, 2))
    ious = intersection / (union + 1e-6)  
    best_idx = np.argmax(ious)
    return ious[best_idx]

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        if not masks_lvl:
            masks_new += ([],) 
            continue
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0, 0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def sam_encoder(mask_generator, image):
    
    # pre-compute masks
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    # pre-compute postprocess
    #masks_default, masks_s, masks_m, masks_l = masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.7, score_thr=0.6, inner_thr=0.5)
   
    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]['segmentation']] = i
        seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
      
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

        return seg_imgs, seg_map
    seg_images, seg_maps = {}, {}
    seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image)
    if len(masks_s) != 0:
        seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image)
    if len(masks_m) != 0:
        seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image)
    if len(masks_l) != 0:
        seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image)

    return seg_images, seg_maps

def get_depth_map(raw_depth):
    raw_depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min() + 1e-8) * 255.0
    raw_depth = raw_depth.astype(np.uint8)
    colorized_depth = torch.cat([raw_depth, raw_depth, raw_depth], dim=0)
    save_image(colorized_depth, '../example/400_depth.png')
    return colorized_depth