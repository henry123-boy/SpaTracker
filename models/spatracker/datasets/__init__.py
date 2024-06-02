# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import numpy as np
import torch 
from torch.utils.data import DataLoader
from models.spatracker.datasets.utils import (
    collate_fn, dataclass_to_cuda_, collate_fn_train
)
from models.spatracker.datasets.tap_vid_datasets import TapVidDataset
from models.spatracker.datasets.badja_dataset import BadjaDataset
from models.spatracker.datasets.fast_capture_dataset import FastCaptureDataset
from models.spatracker.datasets import kubric_movif_dataset
from models.spatracker.datasets import pointodysseydataset_3d
from models.spatracker.datasets import drivetrack_dataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def EvalDatasets(args):
    """
    Returns a list of tuples (dataset_name, dataloader) for all the datasets
    """
    eval_dataloaders = []

    if "badja" in args.eval_datasets:
        eval_dataset = BadjaDataset(
            data_root=os.path.join(args.dataset_root, "BADJA"),
            max_seq_len=args.eval_max_seq_len,
            dataset_resolution=args.crop_size,
        )
        eval_dataloader_badja = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn,
        )
        eval_dataloaders.append(("badja", eval_dataloader_badja))

    if "fastcapture" in args.eval_datasets:
        eval_dataset = FastCaptureDataset(
            data_root=os.path.join(args.dataset_root, "fastcapture"),
            max_seq_len=min(100, args.eval_max_seq_len),
            max_num_points=40,
            dataset_resolution=args.crop_size,
        )
        eval_dataloader_fastcapture = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn,
        )
        eval_dataloaders.append(("fastcapture", eval_dataloader_fastcapture))

    if "tapvid_davis_first" in args.eval_datasets:
        data_root = os.path.join(args.dataset_root, 
                                 "tapvid_davis/tapvid_davis.pkl")
        eval_dataset = TapVidDataset(dataset_type="davis",
                                    data_root=data_root)
        eval_dataloader_tapvid_davis = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn,
        )
        eval_dataloaders.append(("tapvid_davis", eval_dataloader_tapvid_davis)) 

    if 'kubriv' in args.eval_datasets:
        data_root = os.path.join(args.dataset_root, "kubric_eval")
        pass

    return eval_dataloaders   

    
def GetTrainLoader(args, generator=None):
    if 'drivetrack' in args.dataset_root:
        train_dataset = drivetrack_dataset.DriveTrackDataset(
            data_root=os.path.join(args.dataset_root),
            crop_size=args.crop_size,
            seq_len=args.sequence_len,
            traj_per_sample=args.traj_per_sample,
            sample_vis_1st_frame=args.sample_vis_1st_frame,
            use_augs=not args.dont_use_augs,
            use_wind_augs=args.aug_wind_sample,
            use_video_flip=args.use_video_flip,
        )
    else:
        train_dataset = kubric_movif_dataset.KubricMovifDataset(
            data_root=os.path.join(args.dataset_root),
            crop_size=args.crop_size,
            seq_len=args.sequence_len,
            traj_per_sample=args.traj_per_sample,
            sample_vis_1st_frame=args.sample_vis_1st_frame,
            use_augs=not args.dont_use_augs,
            use_wind_augs=args.aug_wind_sample,
            use_video_flip=args.use_video_flip,
            tune_per_scene=args.tune_per_scene,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
        collate_fn=collate_fn_train,
        drop_last=True,
    )
    return train_loader