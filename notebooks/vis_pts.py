import os
import cv2
import numpy as np
import torch
from cotracker.utils.visualizer import Visualizer, read_video_from_path
# ---------- run the spatialtracker ------------
from spatracker1.predictor import CoTrackerPredictor
from spatracker1.zoeDepth.models.builder import build_model
from spatracker1.zoeDepth.utils.config import get_config
from spatracker1.utils.visualizer import Visualizer, read_video_from_path


video = read_video_from_path("./assets/butterfly.mp4")
video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
_, T, C, H, W = video.shape

video = video.cuda()
# init the monocular depth perception
# conf = get_config("zoedepth", "infer", config_version="kitti")
conf = get_config("zoedepth_nk", "infer")
DEVICE = f"cuda:0" if torch.cuda.is_available() else "cpu"
model_zoe_nk = build_model(conf).to(DEVICE)
model_zoe_nk.eval()


import ipdb; ipdb.set_trace()