#python 3.10
"""
    Recover consistent scales and shifts from a set of relative depth input via 
    SpatialTracker

    NOTE: the basic steps includes:
        1. Load the video images 
        2. estimate the depth with relative depth model
        3. recover the consistent scales and shifts via SpatialTracker
        4. optimize the scales and shifts with the estimated depth     
"""

# import the necessary packages
import os, sys
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict

# import the tracker
from models.spatracker_hier.models.build_spatracker import (
    build_spatracker,
)
from models.cotracker.models.core.cotracker.cotracker import (
    get_points_on_a_grid
)
from models.cotracker.utils.visualizer import (
    Visualizer, read_video_from_path
)
from models.cotracker.models.core.model_utils import bilinear_sample2d


# import the monocular depth model
from mde import MonoDEst

class Graph(nn.Module):
    def __init__(self, rel_dp):
        """
            rel_dp: the relative depth map T x 1 x H x W, torch.Tensor
        """
        super(Graph, self).__init__()
        _, T, C, H, W = rel_dp.shape
        self.paras_scale_shift = nn.Parameter(
            torch.ones(T, 2, 1, 1, requires_grad=True)
        )
        self.focal = nn.Parameter(
            512*torch.ones(1, requires_grad=True)
        )
        
        self.rel_dp = rel_dp

    def forward(self, rgbs, tracker, query2d):
        """
        Args:
            rgbs: the input images B x T x 3 x H x W, torch.Tensor
        """
        
        tracker = tracker.cuda().eval()
        rgbds = torch.cat([rgbs, self.rel_dp], dim=2) 
        # the query points
        zeros_one = torch.zeros(query2d.shape[0], query2d.shape[1], 1).cuda()
        query2d_input = torch.cat([zeros_one, query2d], dim=-1)
        # get the metric depth 
        metric_dp = (self.rel_dp * self.paras_scale_shift[None, :, :1, :, :]
                                        + self.paras_scale_shift[None, :, 1:, :, :])

        depth_sample = bilinear_sample2d(
            metric_dp[0, :1], query2d[:, :, 0], query2d[:, :, 1]
        )
        query3d_input = torch.cat([query2d_input, depth_sample.permute(0, 2, 1)], dim=-1)
        tracker.args.depth_near = 0.01
        tracker.args.depth_far = 65
        tracker.args.if_ARAP = True
        tracker.args.Embed3D = True
        with torch.no_grad():
            traj_e, _, vis_e, _ = tracker(rgbds, query3d_input,
                                           iters=4, wind_S=12)
        vis = torch.sigmoid(vis_e)

        depth_est = bilinear_sample2d(
             metric_dp[0, ...], traj_e[0, :, :, 0], traj_e[0, :, :, 1]
            )
        ln = ((depth_est[:,0,:] - traj_e[0,:,:,2])*vis[0]).sum()

        return ln 

# config the shape of the input video and get the queried points
VID_DIR = "./assets/butterfly.mp4"
interp_shape = (384, 512)
grid_pts = get_points_on_a_grid(30, interp_shape)

# config the depth estimator
cfg = edict({
        "mde_name": "depthAny"
        # "mde_name": "zoedepth_nk"
            })
MonoDEst_O = MonoDEst(cfg)
MonoDEst_M = MonoDEst_O.model
MonoDEst_M.eval()

# read video and estimate the relative depth
rgbs = read_video_from_path(VID_DIR)//255
rgbs = torch.from_numpy(rgbs).cuda()[None].permute(0, 1, 4, 2, 3)

rgbs = torch.nn.functional.interpolate(rgbs[0].float(), 
                                       size=interp_shape, mode="bilinear")
T, _, _, _ = rgbs.shape
step_size = 50
rel_dps = None
with torch.no_grad():
    for i in range(0, T, step_size):
        rel_dp = MonoDEst_O.infer(rgbs[i:i+step_size])
        rel_dp = 1/rel_dp
        rel_dp = torch.nn.functional.interpolate(rel_dp, 
                                                size=interp_shape, mode="bilinear")

        if i == 0:
            rel_dps = rel_dp
        else:
            rel_dps = torch.cat([rel_dps, rel_dp], dim=0)

del MonoDEst_O, MonoDEst_M, rel_dp

ckpts = f"checkpoints/spatracker/base/model_spatracker_049500.pth"
model = build_spatracker(ckpts)

# construct the calculating graph
graph = Graph(rel_dps[None]).cuda()

# cfg the optimizer
lr = 1e-3
optimizer = torch.optim.Adam(graph.parameters(), lr=lr)
optim_iters = 10000

for i in range(optim_iters):
    optimizer.zero_grad()
    loss = graph(rgbs[None], model, grid_pts)
    loss.backward()
    optimizer.step()
    print(f"iter {i}, loss {loss.item()}")







