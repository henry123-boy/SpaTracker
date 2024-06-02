
import torch
import torch.nn.functional as F
from base64 import b64encode
import numpy as np
from PIL import Image
import cv2
import argparse
import torchvision.transforms as transforms

#-------- import spatialtracker -------------
from models.spatracker_hier.models.core.spatracker.spatracker import CoTracker


model = CoTracker(
        stride=4,
        S=8,
        add_space_attn=True,
        space_depth=6,
        time_depth=6,
    )

model = model.cuda()

video = torch.randn(1, 50, 4, 384, 512).cuda()
queries = torch.randn(1, 100, 4).cuda()
queries[...,0]=0

model.args.depth_near = 0
model.args.depth_far = 100
model.args.debug = False
model.eval()
import ipdb; ipdb.set_trace()

import time 
time0 = time.time()
output = model(video, queries, iters=4)
time1 = time.time()
print("time: ", time1-time0)
