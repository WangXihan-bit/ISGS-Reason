import torch

def get_ground_instance(xyz: torch.Tensor, instance_ids: torch.Tensor, topk: int = 10):

    assert xyz.shape[0] == instance_ids.shape[0], "The number of points in xyz and instance_ids must be the same."
    
    z = xyz[:, 2] 
    unique_ids = instance_ids.unique()

    id2idx = {uid: i for i, uid in enumerate(unique_ids.tolist())}
    idx_map = torch.tensor([id2idx[i.item()] for i in instance_ids], device=z.device)

    sums = torch.zeros(len(unique_ids), dtype=z.dtype, device=z.device)
    counts = torch.zeros(len(unique_ids), dtype=z.dtype, device=z.device)
    sums.scatter_add_(0, idx_map, z)
    counts.scatter_add_(0, idx_map, torch.ones_like(z))

    mean_heights = sums / (counts + 1e-6)

    topk_counts, topk_indices = torch.topk(counts, k=min(topk, len(unique_ids)))
    candidate_ids = unique_ids[topk_indices]
    candidate_heights = mean_heights[topk_indices]

    min_idx = torch.argmin(candidate_heights)
    ground_instance_id = candidate_ids[min_idx]

    ground_mask = (instance_ids == ground_instance_id)
    return ground_instance_id, ground_mask