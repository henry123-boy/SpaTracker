# %%

#-------- import the base packages -------------
import sys
import os
# os.chdir(os.path.dirname(__file__))
print(f"the current path is: {os.getcwd()}")
sys.path.append(os.getcwd()+"/spatracker1")
sys.path.append(os.getcwd()+"/cotracker")
# print(sys.path)
# print(os.environ)
import torch
import torch.nn.functional as F
from base64 import b64encode
from IPython.display import HTML
import numpy as np
from PIL import Image
import cv2
import argparse

#-------- import cotracker -------------
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

#-------- import spatialtracker -------------
import ipdb; ipdb.set_trace()
from spatracker1.predictor import CoTrackerPredictor
from spatracker1.zoeDepth.models.builder import build_model
from spatracker1.zoeDepth.utils.config import get_config
from spatracker1.utils.visualizer import Visualizer, read_video_from_path

# set the arguments
parser = argparse.ArgumentParser()
# add the video and segmentation
parser.add_argument('--root', type=str, default='./assets', help='path to the video')
parser.add_argument('--vid_name', type=str, default='breakdance', help='path to the video')
# set the gpu
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
# set the model
parser.add_argument('--model', type=str, default='cotracker', help='model name')
# set the downsample factor
parser.add_argument('--downsample', type=int, default=0.8, help='downsample factor')
parser.add_argument('--grid_size', type=int, default=50, help='grid size')
# set the results outdir
parser.add_argument('--outdir', type=str, default='./vis_results', help='output directory')

args = parser.parse_args()

# set input
root_dir = args.root
vid_dir = os.path.join(root_dir, args.vid_name + '.mp4')
seg_dir = os.path.join(root_dir, args.vid_name + '.png')
outdir = args.outdir
os.path.exists(outdir) or os.makedirs(outdir)
# set the paras
grid_size = args.grid_size
model_type = args.model
downsample = args.downsample
# set the gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# read the video
video = read_video_from_path(vid_dir)
video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
_, _, _, H, W = video.shape
# adjust the downsample factor
if H > W:
    downsample = max(downsample, 640//H)
elif H < W:
    downsample = max(downsample, 960//W)
else:
    downsample = max(downsample, 640//H)

video = F.interpolate(video[0], scale_factor=downsample,
                       mode='bilinear', align_corners=True)[None]
import ipdb; ipdb.set_trace()
