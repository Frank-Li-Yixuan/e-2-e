from typing import List, Tuple

import torch
from torchvision.ops import roi_align


def roi_features_from_boxes(feature_map: torch.Tensor, boxes: torch.Tensor, output_size: Tuple[int, int] = (8, 8)) -> torch.Tensor:
    """
    Extract ROI-aligned features from a single feature map given image-space boxes.
    feature_map: [B,C,Hf,Wf] (assume stride s w.r.t. input image)
    boxes: [N,4] in xyxy image coords; if B>1, take first batch (use per-image call)
    returns: [N,C,Oh,Ow]
    """
    if boxes.numel() == 0:
        return torch.zeros((0, feature_map.size(1), *output_size), device=feature_map.device)
    # Build ROI boxes with batch index 0
    b = torch.zeros((boxes.size(0), 1), device=boxes.device)
    rois = torch.cat([b, boxes], dim=1)  # [N,5]
    # Align using spatial scale from fmap to image; if unknown, assume 1 and rely on model's internal scale
    # Practical workaround: roi_align expects rois in same scale as input feature if spatial_scale=1/s.
    # Here we set spatial_scale=feature_map.shape[-1]/W_img; callers should pre-scale boxes if needed.
    return roi_align(feature_map, rois, output_size=output_size, spatial_scale=1.0, aligned=True)


def fuse_global_roi(global_tokens: torch.Tensor, roi_feats: torch.Tensor) -> torch.Tensor:
    """Simple fusion: mean-pool ROI features and concatenate with mean pooled tokens.
    global_tokens: [B,N,C]
    roi_feats: [N,C,Oh,Ow] (N is number of boxes for this image)
    returns [N, Cg + Cr]
    """
    if global_tokens is None or global_tokens.numel() == 0:
        g = torch.zeros((roi_feats.size(0), 0), device=roi_feats.device)
    else:
        gmean = global_tokens.mean(dim=1)  # [B,C]
        g = gmean[:1].expand(roi_feats.size(0), -1)  # [N,C]
    r = roi_feats.mean(dim=[2, 3])  # [N,C]
    return torch.cat([g, r], dim=1)
