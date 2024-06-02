#python3.10 

"""
    Monocular Depth Estimator

    This file contains the Module for Monocular Depth Estimation, including:
        - Midas3.1: https://github.com/isl-org/MiDaS
        - zoedepth: https://github.com/isl-org/ZoeDepth
        - Metric3D: https://github.com/YvanYin/Metric3D
        - Marigold: https://github.com/prs-eth/Marigold
        - Depth-Anything: https://github.com/LiheYoung/Depth-Anything
"""

import os, sys
import importlib
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# import zoedepth builder
from models.monoD.zoeDepth.models.builder import build_model
from models.monoD.zoeDepth.utils.config import get_config
from models.monoD.depth_anything.build import DepthAnything
from easydict import EasyDict as edict

class MonoDEst(nn.Module):

    def __init__(self, args):
        super(MonoDEst, self).__init__()

        # build the chosen model
        if args.mde_name == "zoedepth_nk":
            conf = get_config("zoedepth_nk", "infer")
            model_zoe_nk = build_model(conf)
            model_zoe_nk.eval()
            model_zoe_nk = model_zoe_nk.cuda()
            self.model = model_zoe_nk
        elif args.mde_name == "zoedepth_k":
            conf = get_config("zoedepth", "infer", config_version="kitti")
            model_zoe_k = build_model(conf)
            model_zoe_k.eval()
            model_zoe_k = model_zoe_k.cuda()
            self.model = model_zoe_k
        elif args.mde_name == "depthAny":
            cfg = edict({
                "encoder": "vits",
                "load_from": "models/monoD/depth_anything/ckpts/depth_anything_vits14.pth",
                "localhub": True
            })
            self.model = DepthAnything(cfg)
            # get one metric model
            conf = get_config("zoedepth_nk", "infer")
            model_zoe_nk = build_model(conf)
            model_zoe_nk.eval()
            model_zoe_nk = model_zoe_nk.cuda()
            self.metric3d = model_zoe_nk
        self.mde_name = args.mde_name

    def infer(self, rgbs, scale=None, shift=None):
        """
            Infer the depth map from the input RGB image
        """

        # get the depth map
        if self.mde_name == "depthAny":
            depth_map = self.model.infer(rgbs)
            metric_dp = self.metric3d.infer(rgbs[:20])
            metric_dp_inv = 1/metric_dp
            dp_0_rel = depth_map[:20]
            scale,shift = np.polyfit(dp_0_rel.view(-1).cpu().numpy(),
                             metric_dp_inv.view(-1).cpu().numpy(), 1)
            depth_map = depth_map*scale + shift
            depth_map = (1/depth_map).clamp(0.01, 65)
        else:
            depth_map = self.model.infer(rgbs)

        return depth_map
    

def write_ply(points,colors,path_ply,mask=None):
    if mask is not None:
        num = np.sum(mask)
    else:
        num = points.shape[0]
    ply_header = '''ply
        format ascii 1.0
        element vertex {}
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''.format(num)
    if mask is not None:
        with open(path_ply+'_mask'+'.ply', 'w') as f:
            f.write(ply_header)
            for i in range(points.shape[0]):
                if mask.reshape(-1)[i]:
                    f.write('{} {} {} {} {} {}\n'.format(points[i,0], points[i,1], points[i,2],
                                                                    int(colors[i, 0]*255), int(colors[i, 1]*255), int(colors[i, 2]*255)))
    else:
        with open(path_ply+'.ply', 'w') as f:
            f.write(ply_header)
            for i in range(points.shape[0]):
                f.write('{} {} {} {} {} {}\n'.format(points[i,0], points[i,1], points[i,2],
                                                                    int(colors[i, 0]*255), int(colors[i, 1]*255), int(colors[i, 2]*255)))

#TODO: unit test
if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    import imageio
    def pixel_to_focal(pixels_a, K_a):
        # project pixels_b to 3D points (x, y, z) in cam_a coordinates
        points_a_cam = np.linalg.inv(K_a) @ pixels_a
        return points_a_cam

    def focal_to_camera(points_a_cam, depth_a):
        points_a_cam *= depth_a.flatten()
        return points_a_cam

    def get_pixel(H, W):
        # get 2D pixels (u, v) for image_a in cam_a pixel space
        u_a, v_a = np.meshgrid(np.arange(W), np.arange(H))
        u_a = np.flip(u_a, axis=1)
        v_a = np.flip(v_a, axis=0)
        pixels_a = np.stack([
            u_a.flatten() + 0.5, 
            v_a.flatten() + 0.5, 
            np.ones_like(u_a.flatten())
        ], axis=0)
        
        return pixels_a

    def get_intrinsics(H, W):
        """
        Intrinsics for a pinhole camera model.
        Assume fov of 55 degrees and central principal point.
        """
        f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
        cx = 0.5 * W
        cy = 0.5 * H
        return np.array([[f, 0, cx],
                        [0, f, cy],
                        [0, 0, 1]])

    cfg = edict({
        "encoder": "vits",
        "load_from": "models/monoD/depth_anything/ckpts/depth_anything_vits14.pth",
        "localhub": True
    })
    model = DepthAnything(cfg)
    DATA_ROOT = "/nas2/xyx/kubric/movi_f/"
    SCENE_NUM = "0"
    FRAME_NUM = f"{int(0):05d}"
    # img_np = cv2.imread(os.path.join(DATA_ROOT,"512x512_frames", SCENE_NUM, FRAME_NUM+".jpeg"))
    img_np = cv2.imread("/nas2/xyx/BADJA/BADJA/DAVIS/JPEGImages/Full-Resolution/car-roundabout/00000.jpg")
    img = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().cuda()
    # load the RGB image
    depth = model.infer(img/255).detach().cpu().numpy()
    depth = (depth-depth.min())/(depth.max()-depth.min())

    # depth from depth anything 
    H, W, _ = img_np.shape
    K = get_intrinsics(H, W)
    factor = H // 1
    pixels = get_pixel(H, W)/factor
    # focals = pixel_to_focal(pixels, K)
    # points = focal_to_camera(focals, depth)

    points = pixels.transpose(1, 0)
    points[:,2]=depth.reshape(-1)
    colors = img_np.reshape(-1, 3) / 255.0
    write_ply(points, colors, 'depthAny.ply')

    # depth from the gt depth
    depth_gt = imageio.v2.imread(os.path.join(DATA_ROOT, "512x512_depth", SCENE_NUM, FRAME_NUM+".png"))/1000
    # points = focal_to_camera(focals, depth_gt/1000)
    min = depth_gt[depth_gt>0].min()
    depth_gt_inv = 1/(depth_gt.clip(min, 65))
    points = pixels.transpose(1, 0)
    points[:,2]=depth_gt.reshape(-1)
    colors = img_np.reshape(-1, 3) / 255.0
    write_ply(points, colors, 'depthGT.py')

    # depth of align 
    scale, shift = np.polyfit(depth.reshape(-1),depth_gt_inv.reshape(-1), 1)
    depth_align = depth*scale + shift
    points = pixels.transpose(1, 0)
    points[:,2]=1/depth_align.reshape(-1)
    colors = img_np.reshape(-1, 3) / 255.0
    write_ply(points, colors, 'depthAlign.py')
