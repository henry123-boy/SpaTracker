from numpy import random
import torch
import numpy as np
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
from torch._C import dtype, set_flush_denormal
import models.spatracker.utils.basic
import models.spatracker.utils.improc
import models.spatracker.utils.geom as geom
import models.spatracker.utils.misc as misc
from models.spatracker.datasets.utils import CoTrackerData, aug_depth


import glob
import cv2
from torchvision.transforms import ColorJitter, GaussianBlur
import albumentations as A
from functools import partial
import sys

# np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

class PointOdysseyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_location='/orion/group/point_odyssey',
                 dset='train',
                 use_augs=False,
                 S=8,
                 N=32,
                 video_len=24,
                 strides=[1],
                 quick=False,
                 verbose=False,
    ):
        print('loading pointodyssey dataset...')
        # S is the number of frames in a clip
        self.S = S
        self.seq_len = S
        self.N = N

        self.use_augs = use_augs
        self.dset = dset

        self.rgb_paths = []
        self.depth_paths = []
        self.normal_paths = []
        self.traj_paths = []
        self.annotation_paths = []
        self.full_idxs = []

        self.subdirs = []
        self.sequences = []
        self.subdirs.append(os.path.join(dataset_location, dset))

        for subdir in self.subdirs:
            for seq in glob.glob(os.path.join(subdir, "*/")):
                seq_name = seq.split('/')[-1]
                self.sequences.append(seq)
        
        self.sequences = sorted(self.sequences)
        if verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))
        
        ## load trajectories
        print('loading trajectories...')

        if quick:
           self.sequences = self.sequences[:1] 
        
        for seq in self.sequences:

            # if 'character1' not in seq:
            rgb_path = os.path.join(seq, 'rgbs')

            annotations_path = os.path.join(seq, 'annot.npz')
            if os.path.isfile(annotations_path):

                if verbose: 
                    print('seq', seq)
                    
                for stride in strides:
                    # split the data with the fixed stride
                    for ii in range(0,len(os.listdir(rgb_path))-self.S*stride+1, self.S//2):
                        full_idx = ii + np.arange(self.S)*stride
                        self.rgb_paths.append([os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % idx) for idx in full_idx])
                        self.depth_paths.append([os.path.join(seq, 'depths', 'depth_%05d.png' % idx) for idx in full_idx])
                        self.normal_paths.append([os.path.join(seq, 'normals', 'normal_%05d.jpg' % idx) for idx in full_idx])
                        self.annotation_paths.append(os.path.join(seq, 'annot.npz'))
                        self.full_idxs.append(full_idx)
                        if verbose:
                            sys.stdout.write('.')
                            sys.stdout.flush()
                        else:
                            if verbose:
                                sys.stdout.write('v')
                                sys.stdout.flush()

        print('collected %d clips of length %d in %s (dset=%s)' % (
            len(self.rgb_paths), self.S, dataset_location, dset))

    def getitem_helper(self, index):
        sample = None
        gotit = False
        rgb_paths = self.rgb_paths[index]
        depth_paths = self.depth_paths[index]
        normal_paths = self.normal_paths[index]
        full_idx = self.full_idxs[index]
        annotations_path = self.annotation_paths[index]
        annotations = np.load(annotations_path, allow_pickle=True)
        trajs_2d = annotations['trajs_2d'][full_idx].astype(np.float32)
        visibs = annotations['visibs'][full_idx].astype(np.float32)
        valids = (visibs<2).astype(np.float32)
        visibs = (visibs==1).astype(np.float32)
        trajs_world = annotations['trajs_3d'][full_idx].astype(np.float32)
        pix_T_cams = annotations['intrinsics'][full_idx].astype(np.float32)
        cams_T_world = annotations['extrinsics'][full_idx].astype(np.float32)

        # ensure that the point is good in frame0
        vis_and_val = valids * visibs
        vis0 = vis_and_val[0] > 0
        trajs_2d = trajs_2d[:,vis0]
        trajs_world = trajs_world[:,vis0]
        visibs = visibs[:,vis0]
        valids = valids[:,vis0]

        S,N,D = trajs_2d.shape
        assert(D==2)
        assert(S==self.S)
        
        if N < self.N//2:
            print('returning before cropping: N=%d; need at least N=%d' % (N, self.N//2))
            return None, False
        
        trajs_cam = geom.apply_4x4_py(cams_T_world, trajs_world)
        trajs_pix = geom.apply_pix_T_cam_py(pix_T_cams, trajs_cam)

        # get rid of infs and nans
        valids_xy = np.ones_like(trajs_2d)
        inf_idx = np.where(np.isinf(trajs_2d))
        trajs_2d[inf_idx] = 0
        valids_xy[inf_idx] = 0
        nan_idx = np.where(np.isnan(trajs_2d))
        trajs_2d[nan_idx] = 0
        valids_xy[nan_idx] = 0
        inv_idx = np.where(np.sum(valids_xy, axis=2)<2) # S,N
        visibs[inv_idx] = 0
        valids[inv_idx] = 0

        rgbs = []
        for rgb_path in rgb_paths:
            with Image.open(rgb_path) as im:
                rgbs.append(np.array(im)[:, :, :3])

        depths = []
        for depth_path in depth_paths:
            depth16 = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            depth = depth16.astype(np.float32) / 65535.0 * 1000.0
            depths.append(depth)

        normals = []
        for normal_path in normal_paths:
            with Image.open(normal_path) as im:
                normals.append(np.array(im)[:, :, :3])
            

        H,W,C = rgbs[0].shape
        assert(C==3)
        
        # update visibility annotations
        for si in range(S):
            # avoid 1px edge
            oob_inds = np.logical_or(
                np.logical_or(trajs_2d[si,:,0] < 1, trajs_2d[si,:,0] > W-2),
                np.logical_or(trajs_2d[si,:,1] < 1, trajs_2d[si,:,1] > H-2))
            visibs[si,oob_inds] = 0

            # when a point moves far oob, don't supervise with it
            very_oob_inds = np.logical_or(
                np.logical_or(trajs_2d[si,:,0] < -64, trajs_2d[si,:,0] > W+64),
                np.logical_or(trajs_2d[si,:,1] < -64, trajs_2d[si,:,1] > H+64))
            valids[si,very_oob_inds] = 0

        # ensure that the point is good in frame0
        vis_and_val = valids * visibs
        vis0 = vis_and_val[0] > 0
        trajs_2d = trajs_2d[:,vis0]
        trajs_cam = trajs_cam[:,vis0]
        trajs_pix = trajs_pix[:,vis0]
        visibs = visibs[:,vis0]
        valids = valids[:,vis0]

        # ensure that the point is good in frame1
        vis_and_val = valids * visibs
        vis1 = vis_and_val[1] > 0
        trajs_2d = trajs_2d[:,vis1]
        trajs_cam = trajs_cam[:,vis1]
        trajs_pix = trajs_pix[:,vis1]
        visibs = visibs[:,vis1]
        valids = valids[:,vis1]

        # ensure that the point is visible at frame0
        vis0 = visibs[0] > 0
        trajs_2d = trajs_2d[:,vis0]
        trajs_cam = trajs_cam[:,vis0]
        trajs_pix = trajs_pix[:,vis0]
        visibs = visibs[:,vis0]
        valids = valids[:,vis0]

        # ensure that the point is good in at least sqrt(S) frames
        val_ok = np.sum(valids, axis=0) >= max(np.sqrt(S),2)
        trajs_2d = trajs_2d[:,val_ok]
        trajs_cam = trajs_cam[:,val_ok]
        trajs_pix = trajs_pix[:,val_ok]
        visibs = visibs[:,val_ok]
        valids = valids[:,val_ok]
        
        N = trajs_2d.shape[1]
        
        if N < self.N//2:
            # print('N=%d' % (N))
            return None, False
        
        if N < self.N:
            print('N=%d; ideally we want N=%d, but we will pad' % (N, self.N))

        if N > self.N*4:
            # fps based on position and motion
            xym = np.concatenate([np.mean(trajs_2d, axis=0), np.mean(trajs_2d[1:] - trajs_2d[:-1], axis=0)], axis=-1)
            inds = misc.farthest_point_sample_py(xym, self.N*4)
            trajs_2d = trajs_2d[:,inds]
            trajs_cam = trajs_cam[:,inds]
            trajs_pix = trajs_pix[:,inds]
            visibs = visibs[:,inds]
            valids = valids[:,inds]

        # clamp so that the trajectories don't get too crazy
        trajs_2d = np.minimum(np.maximum(trajs_2d, np.array([-64,-64])), np.array([W+64, H+64])) # S,N,2
        trajs_pix = np.minimum(np.maximum(trajs_pix, np.array([-64,-64])), np.array([W+64, H+64])) # S,N,2
            
        N = trajs_2d.shape[1]
        N_ = min(N, self.N)
        inds = np.random.choice(N, N_, replace=False)

        # prep for batching, by fixing N
        trajs_2d_full = np.zeros((self.S, self.N, 2)).astype(np.float32)
        trajs_cam_full = np.zeros((self.S, self.N, 3)).astype(np.float32)
        trajs_pix_full = np.zeros((self.S, self.N, 2)).astype(np.float32)
        visibs_full = np.zeros((self.S, self.N)).astype(np.float32)
        valids_full = np.zeros((self.S, self.N)).astype(np.float32)
        trajs_2d_full[:,:N_] = trajs_2d[:,inds]
        trajs_cam_full[:,:N_] = trajs_cam[:,inds]
        trajs_pix_full[:,:N_] = trajs_pix[:,inds]
        visibs_full[:,:N_] = visibs[:,inds]
        valids_full[:,:N_] = valids[:,inds]

        rgbs = torch.from_numpy(np.stack(rgbs, 0)).permute(0,3,1,2) # S,3,H,W
        depths = torch.from_numpy(np.stack(depths, 0)).unsqueeze(1) # S,1,H,W
        normals = torch.from_numpy(np.stack(normals, 0)).permute(0,3,1,2) # S,3,H,W
        trajs_2d = torch.from_numpy(trajs_2d_full) # S,N,2
        trajs_cam = torch.from_numpy(trajs_cam_full) # S,N,3
        trajs_pix = torch.from_numpy(trajs_pix_full) # S,N,2
        visibs = torch.from_numpy(visibs_full) # S,N
        valids = torch.from_numpy(valids_full) # S,N

        # sample = {
        #     'rgbs': rgbs,
        #     'depths': depths,
        #     'normals': normals,
        #     'trajs_2d': trajs_2d,
        #     'trajs_cam': trajs_cam,
        #     'trajs_pix': trajs_pix,
        #     'pix_T_cams': pix_T_cams,
        #     'cams_T_world': cams_T_world,
        #     'visibs': visibs,
        #     'valids': valids,
        # }

        segs = torch.ones((self.seq_len, 1, 
                                depths.shape[2], depths.shape[3]))

        sample = CoTrackerData(
            video=rgbs,
            videodepth=depths,
            segmentation=segs,
            trajectory=trajs_2d,
            visibility=visibs,
            valid=valids,
            seq_name=f"{full_idx}",
        )

        return sample, True

    
    def __getitem__(self, index):
        gotit = False
        
        sample, gotit = self.getitem_helper(index)
        if not gotit:
            print("warning: sampling failed")
            # fake sample, so we can still collate
            sample = CoTrackerData(
                video=torch.zeros(
                    (self.seq_len, 3, self.crop_size[0], self.crop_size[1])
                ),
                videodepth=torch.zeros(
                    (self.seq_len, 1, self.crop_size[0], self.crop_size[1])
                ),
                segmentation=torch.zeros(
                    (self.seq_len, 1, self.crop_size[0], self.crop_size[1])
                ),
                trajectory=torch.zeros((self.seq_len, self.traj_per_sample, 2)),
                visibility=torch.zeros((self.seq_len, self.traj_per_sample)),
                valid=torch.zeros((self.seq_len, self.traj_per_sample)),
            )
        return sample, gotit

    def __len__(self):
        return len(self.rgb_paths)
    

# unit test
if __name__ == '__main__':
    dataset_pod = PointOdysseyDataset(
                    dataset_location='/nas2/xyx/point_odyssey',
                    dset='train',
                    use_augs=False,
                    S=56,
                    N=256,
                    strides=[1],
                    quick=False,
                    verbose=False,
                )
