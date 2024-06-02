import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from models.monoD.depth_anything.dpt import DPT_DINOv2
from models.monoD.depth_anything.util.transform import (
    Resize, NormalizeImage, PrepareForNet
)


def build(config):
    """
        Build the model from the config
        NOTE: the config should contain the following
        - encoder: the encoder type of the model
        - load_from: the path to the pretrained model
    """
    args = config
    assert args.encoder in ['vits', 'vitb', 'vitl']
    if args.encoder == 'vits':
        depth_anything = DPT_DINOv2(encoder='vits', features=64,
                                           out_channels=[48, 96, 192, 384],
                                            localhub=args.localhub).cuda()
    elif args.encoder == 'vitb':
        depth_anything = DPT_DINOv2(encoder='vitb', features=128,
                                     out_channels=[96, 192, 384, 768],
                                       localhub=args.localhub).cuda()
    else:
        depth_anything = DPT_DINOv2(encoder='vitl', features=256,
                                     out_channels=[256, 512, 1024, 1024],
                                       localhub=args.localhub).cuda()
    depth_anything.load_state_dict(torch.load(args.load_from,
                                               map_location='cpu'), strict=True)
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))    
    depth_anything.eval()

    return depth_anything

class DepthAnything(nn.Module):
    def __init__(self, args):
        super(DepthAnything, self).__init__()

        # build the chosen model
        self.dpAny = build(args)

    def infer(self, rgbs):
        """
            Infer the depth map from the input RGB image
        
        Args:
            rgbs: the input RGB image B x 3 x H x W (Cuda Tensor)
        
        Asserts:
            the input should be a cuda tensor
        """
        assert (rgbs.is_cuda)&(len(rgbs.shape) == 4)
        T, C, H, W = rgbs.shape
        # prepare the input
        Resizer = Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            )       
        #NOTE: step 1 Resize 
        width, height = Resizer.get_size( 
            rgbs.shape[2], rgbs.shape[3]
        )
        rgbs = F.interpolate(
            rgbs, (int(height), int(width)), mode='bicubic', align_corners=False
        )
        #NOTE: step 2 NormalizeImage
        mean_ = torch.tensor([0.485, 0.456, 0.406], 
                             device=rgbs.device).view(1, 3, 1, 1)
        std_ = torch.tensor([0.229, 0.224, 0.225],
                            device=rgbs.device).view(1, 3, 1, 1)
        rgbs = (rgbs - mean_)/std_
        #NOTE: step 3 PrepareForNet

        # get the depth map

        disp = self.dpAny(rgbs)
        disp = F.interpolate(
            disp[:,None], (H, W), 
            mode='bilinear', align_corners=False
        )
        # clamping the farthest depth to 100x of the nearest
        depth_map = disp        

        return depth_map

