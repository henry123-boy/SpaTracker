# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# %%

import torch
import dataclasses
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(eq=False)
class CoTrackerData:
    """
    Dataclass for storing video tracks data.
    """

    video: torch.Tensor  # B, S, C, H, W
    segmentation: torch.Tensor  # B, S, 1, H, W
    trajectory: torch.Tensor  # B, S, N, 2
    visibility: torch.Tensor  # B, S, N
    # optional data
    videodepth: Optional[torch.Tensor] = None # B, S, 1, H, W
    valid: Optional[torch.Tensor] = None  # B, S, N
    seq_name: Optional[str] = None
    intrs: Optional[torch.Tensor] = torch.eye(3).unsqueeze(0)  # B, 3, 3
    query_points: Optional[torch.Tensor] = None  # TapVID evaluation format


def collate_fn(batch):
    """
    Collate function for video tracks data.
    """
    video = torch.stack([b.video for b in batch], dim=0)
    # videodepth = torch.stack([b.videodepth for b in batch], dim=0)
    segmentation = torch.stack([b.segmentation for b in batch], dim=0)
    trajectory = torch.stack([b.trajectory for b in batch], dim=0)
    visibility = torch.stack([b.visibility for b in batch], dim=0)
    query_points = None
    if batch[0].query_points is not None:
        query_points = torch.stack([b.query_points for b in batch], dim=0)
    seq_name = [b.seq_name for b in batch]
    intrs = torch.stack([b.intrs for b in batch],dim=0)

    return CoTrackerData(
        video=video,
        segmentation=segmentation,
        trajectory=trajectory,
        visibility=visibility,
        seq_name=seq_name,
        query_points=query_points,
        intrs=intrs,
    )


def collate_fn_train(batch):
    """
    Collate function for video tracks data during training.
    """
    gotit = [gotit for _, gotit in batch]
    video = torch.stack([b.video for b, _ in batch], dim=0)
    videodepth = torch.stack([b.videodepth for b, _ in batch], dim=0)
    segmentation = torch.stack([b.segmentation for b, _ in batch], dim=0)
    trajectory = torch.stack([b.trajectory for b, _ in batch], dim=0)
    visibility = torch.stack([b.visibility for b, _ in batch], dim=0)
    valid = torch.stack([b.valid for b, _ in batch], dim=0)
    seq_name = [b.seq_name for b, _ in batch]
    intrs = torch.stack([b.intrs for b, _ in batch],dim=0)
    return (
        CoTrackerData(video=video, 
                      videodepth=videodepth,
                      segmentation=segmentation,
                      trajectory=trajectory, 
                      visibility=visibility, valid=valid,
                      seq_name=seq_name,
                      intrs=intrs),
        gotit,
    )


def try_to_cuda(t: Any) -> Any:
    """
    Try to move the input variable `t` to a cuda device.

    Args:
        t: Input.

    Returns:
        t_cuda: `t` moved to a cuda device, if supported.
    """
    try:
        t = t.float().cuda()
    except AttributeError:
        pass
    return t


def dataclass_to_cuda_(obj):
    """
    Move all contents of a dataclass to cuda inplace if supported.

    Args:
        batch: Input dataclass.

    Returns:
        batch_cuda: `batch` moved to a cuda device, if supported.
    """
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cuda(getattr(obj, f.name)))
    return obj


def resize_sample(rgbs, trajs_g, segs, interp_shape):
    S, C, H, W = rgbs.shape
    S, N, D = trajs_g.shape

    assert D == 2

    rgbs = F.interpolate(rgbs, interp_shape, mode="bilinear")
    segs = F.interpolate(segs, interp_shape, mode="nearest")

    trajs_g[:, :, 0] *= interp_shape[1] / W
    trajs_g[:, :, 1] *= interp_shape[0] / H
    return rgbs, trajs_g, segs



def aug_depth(depth,
              grid: tuple = (8, 8),
              scale: tuple = (0.7, 1.3),
              shift: tuple = (-0.1, 0.1),
              gn_kernel: tuple = (7, 7),
              gn_sigma: tuple = (2.0, 2.0)
              ):
    """
    Augment depth for training. 
    Include:
        - low resolution scale and shift augmentation
        - gaussian blurring with random kernel size
    
    Args:
        depth: 1 T H W tensor
        grid: resolution for scale and shift augmentation 16 * 16 by default

    """
    
    B, T, H, W = depth.shape
    msk = (depth != 0)
    # generate the scale and shift map 
    H_s, W_s = grid
    scale_map = (torch.rand(B, T, 
                           H_s, W_s, 
                           device=depth.device) * (scale[1] - scale[0])
                             + scale[0] )
    shift_map = (torch.rand(B, T,
                            H_s, W_s, 
                            device=depth.device) * (shift[1] - shift[0])
                              + shift[0] )
    
    # scale and shift the depth map
    scale_map = F.interpolate(scale_map, (H, W), 
                              mode='bilinear', align_corners=True)
    shift_map = F.interpolate(shift_map, (H, W), 
                              mode='bilinear', align_corners=True)

    # local scale and shift the depth
    depth[msk] = (depth[msk] * scale_map[msk])+ shift_map[msk]*(depth[msk].mean())

    # gaussian blur
    depth = TF.gaussian_blur(depth, kernel_size=gn_kernel, sigma=gn_sigma)
    depth[~msk] = 0

    return depth




# unit test
if __name__ == "__main__":
    import os
    import cv2 as cv
    import matplotlib.pyplot as plt
    import numpy as np
    import tripy
    
    # ----------- test the depth augmentation -----------------
    dp_path_root = "/nas2/xyx/point_odyssey/train/ani"
    dp_path = os.path.join(dp_path_root, "depths")
    dp_path_list = sorted(os.listdir(dp_path))
    dp_path_0 = os.path.join(dp_path, dp_path_list[500])
    # get the depth 
    depth16 = cv.imread(dp_path_0, cv.IMREAD_ANYDEPTH)
    depth = depth16.astype(np.float32) / 65535.0 * 1000.0
    depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
    depth[depth == 0] = 64
    depth_aug = aug_depth(depth,
                    grid=(8, 8),
                    scale=(0.7, 1.3),
                    shift=(-0.1, 0.1),
                        )
    plt.subplot(1, 2, 1)
    plt.imshow(depth_aug.squeeze().numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(depth.squeeze().numpy())
    plt.show()
    print(depth.shape)
    # ---------------------------------------------------------
    
    # ----------- test the 3D point cloud -------------------
    annot_path = os.path.join(dp_path_root, "annot.npz")
    annot = np.load(annot_path)
    intr = annot['intrinsics'][0]
    x = torch.arange(0, depth.shape[3], 1, dtype=torch.float32)
    y = torch.arange(0, depth.shape[2], 1, dtype=torch.float32)
    x, y = torch.meshgrid(x, y)
    depth_vis = depth.squeeze() # H, W
    xy = torch.stack([x, y], dim=-1) # W, H, 2
    z = depth_vis.transpose(1,0) # W, H
    # unproj 
    xy[..., 0] = (xy[..., 0] - intr[0, 2]) / intr[0, 0]
    xy[..., 1] = (xy[..., 1] - intr[1, 2]) / intr[1, 1]
    xyz = torch.cat([z[..., None]*xy, z[..., None]], dim=-1) # W, H, 3
    # 
    from plyfile import PlyData, PlyElement
    pcd = xyz.view(-1, 3).cpu().numpy()
    vertex = np.core.records.fromarrays(pcd.transpose(), names='x, y, z')
    vertices = PlyElement.describe(vertex, 'vertex')
    plydata = PlyData([vertices])
    plydata.write(f'pcdtest.ply')

    import ipdb; ipdb.set_trace()
