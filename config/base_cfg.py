#python3.10
"""Hierachical configuration for different pipelines, using `yacs` 
(refered to https://github.com/rbgirshick/yacs) 

This projects contain the configuration for three aspects: 
    the regular config for experiment setting

    NOTE: Each experiment will be assigned a seperate working space, and the 
    intermediate results will be saved in the working space. The experimentes 
    folder structure is as follows:
    {
        /${ROOT_WORK_DIR}/
        └── ${PIPELINES_NAME}/
            ├── ${EXP_NAME}/
                ├── ${CHECKPOINT_DIR}/
                ├── ${RESULT_DIR}/
                ├── meta.json/
                └── ${LOG_DIR} 
    }

"""

import os, sys
from .yacs import CfgNode as CN
import argparse
import numpy as np

# the parser for boolean
def bool_parser(arg):
    """Parses an argument to boolean."""
    if isinstance(arg, bool):
        return arg
    if arg is None:
        return False
    if arg.lower() in ['1', 'true', 't', 'yes', 'y']:
        return True
    if arg.lower() in ['0', 'false', 'f', 'no', 'n']:
        return False
    raise ValueError(f'`{arg}` cannot be converted to boolean!')

# -----------------------------------------------------------------------------
# base cfg
# -----------------------------------------------------------------------------
cfg = CN()

# configuration for basic experiments
cfg.save_dir = "./checkpoints"
cfg.restore_ckpt = ""
cfg.model_name = "cotracker"
cfg.exp_name = ""

# NOTE: configuration for datasets and augmentation
cfg.dataset_root = ""
cfg.eval_datasets = [""]
cfg.dont_use_augs = False
cfg.crop_size = [384, 512]
cfg.traj_per_sample = 384
cfg.sample_vis_1st_frame = False
cfg.depth_near = 0.01 # meter
cfg.depth_far = 65.0 # meter
cfg.sequence_len = 24

# NOTE: configuration for network arch
cfg.sliding_window_len = 8
cfg.remove_space_attn = False
cfg.updateformer_hidden_size = 384
cfg.updateformer_num_heads = 8
cfg.updateformer_space_depth = 6
cfg.updateformer_time_depth = 6
cfg.model_stride = 4
cfg.train_iters = 4
cfg.if_ARAP = False
cfg.Embed3D = False
cfg.Loss_W_feat = 5e-1
cfg.Loss_W_cls = 1e-4
cfg.depth_color = False
cfg.flash_attn = False
cfg.corr_dp = True
cfg.support_grid = 0
cfg.backbone = "CNN"
cfg.enc_only = False
cfg.init_match = False
cfg.Nblock = 4

# NOTE: configuration for training and saving
cfg.nodes_num = 1
cfg.batch_size = 1
cfg.num_workers = 6
cfg.mixed_precision = False
cfg.lr = 0.0005
cfg.wdecay = 0.00001
cfg.num_steps = 200000
cfg.evaluate_every_n_epoch = 1
cfg.save_every_n_epoch = 1
cfg.validate_at_start = False
cfg.save_freq = 100
cfg.eval_max_seq_len = 1000
cfg.debug = False
cfg.fine_tune = False
cfg.aug_wind_sample = False
cfg.use_video_flip = False
cfg.fix_backbone = False
cfg.tune_backbone = False
cfg.tune_arap = False
cfg.tune_per_scene = False
cfg.use_hier_encoder = False
cfg.scales = [4, 2]


# NOTE: configuration for monocular depth estimator
cfg.mde_name = "zoedepth_nk"

# -----------------------------------------------------------------------------

# configurations for the command line
parser = argparse.ArgumentParser()

# config for the basic experiment
parser.add_argument("--save_dir", default="./checkpoints", type=str ,help="path to save checkpoints")
parser.add_argument("--restore_ckpt", default="", help="path to restore a checkpoint")
parser.add_argument("--model_name", default="cotracker", help="model name")
parser.add_argument("--exp_name", type=str, default="base",
                    help="the name for experiment",
                    )
# config for dataset and augmentation
parser.add_argument(
    "--dataset_root", type=str, help="path lo all the datasets (train and eval)"
)
parser.add_argument(
    "--eval_datasets", nargs="+", default=["things", "badja"],
    help="what datasets to use for evaluation",
)
parser.add_argument(
    "--dont_use_augs", action="store_true", default=False,
    help="don't apply augmentations during training",
)
parser.add_argument(
    "--crop_size", type=int, nargs="+", default=[384, 512],
    help="crop videos to this resolution during training",
)
parser.add_argument(
    "--traj_per_sample", type=int, default=768,
    help="the number of trajectories to sample for training",
)
parser.add_argument(
    "--depth_near", type=float, default=0.01, help="near plane depth"
)
parser.add_argument(
    "--depth_far", type=float, default=65.0, help="far plane depth"
)
parser.add_argument(
    "--sample_vis_1st_frame",
    action="store_true",
    default=False,
    help="only sample trajectories with points visible on the first frame",
)
parser.add_argument(
    "--sequence_len", type=int, default=24, help="train sequence length"
)
# configuration for network arch
parser.add_argument(
    "--sliding_window_len",
    type=int,
    default=8,
    help="length of the CoTracker sliding window",
)
parser.add_argument(
    "--remove_space_attn",
    action="store_true",
    default=False,
    help="remove space attention from CoTracker",
)
parser.add_argument(
    "--updateformer_hidden_size",
    type=int,
    default=384,
    help="hidden dimension of the CoTracker transformer model",
)
parser.add_argument(
    "--updateformer_num_heads",
    type=int,
    default=8,
    help="number of heads of the CoTracker transformer model",
)
parser.add_argument(
    "--updateformer_space_depth",
    type=int,
    default=6,
    help="number of group attention layers in the CoTracker transformer model",
)
parser.add_argument(
    "--updateformer_time_depth",
    type=int,
    default=6,
    help="number of time attention layers in the CoTracker transformer model",
)
parser.add_argument(
    "--model_stride",
    type=int,
    default=4,
    help="stride of the CoTracker feature network",
)
parser.add_argument(
    "--train_iters",
    type=int,
    default=4,
    help="number of updates to the disparity field in each forward pass.",
)
parser.add_argument(
    "--if_ARAP",
    action="store_true",
    default=False,
    help="if using ARAP loss in the optimization",
)
parser.add_argument(
    "--Embed3D",
    action="store_true",
    default=False,
    help="if using the 3D embedding for image",
)
parser.add_argument(
    "--Loss_W_feat",
    type=float,
    default=5e-1,
    help="weight for the feature loss",
)
parser.add_argument(
    "--Loss_W_cls",
    type=float,
    default=1e-4,
    help="weight for the classification loss",
)
parser.add_argument(
    "--depth_color",
    action="store_true",
    default=False,
    help="if using the color for depth",
)
parser.add_argument(
    "--flash_attn",
    action="store_true",
    default=False,
    help="if using the flash attention",
)
parser.add_argument(
    "--corr_dp",
    action="store_true",
    default=False,
    help="if using the correlation of depth",
)
parser.add_argument(
    "--support_grid",
    type=int,
    default=0,
    help="if using the support grid",
)
parser.add_argument(
    "--backbone",
    type=str,
    default="CNN",
    help="backbone for the CoTracker feature network",
)
parser.add_argument(
    "--enc_only",
    action="store_true",
    default=False,
    help="if using the encoder only",
)
parser.add_argument(
    "--init_match",
    action="store_true",
    default=False,
    help="if using the initial matching",
)
parser.add_argument(
    "--Nblock",
    type=int,
    default=4,
    help="number of blocks in the CoTracker feature network",
)

# configuration for training and saving
parser.add_argument(
    "--nodes_num", type=int, default=1, help="number of nodes used for training."
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="batch size used during training."
)
parser.add_argument(
    "--num_workers", type=int, default=6, help="number of dataloader workers"
)

parser.add_argument(
    "--mixed_precision", 
    action="store_true", default=False,
    help="use mixed precision"
)
parser.add_argument("--lr", type=float, default=0.0005, help="max learning rate.")
parser.add_argument(
    "--wdecay", type=float, default=0.00001, help="Weight decay in optimizer."
)
parser.add_argument(
    "--num_steps", type=int, default=200000, help="length of training schedule."
)
parser.add_argument(
    "--evaluate_every_n_epoch",
    type=int,
    default=1,
    help="evaluate during training after every n epochs, after every epoch by default",
)
parser.add_argument(
    "--save_every_n_epoch",
    type=int,
    default=1,
    help="save checkpoints during training after every n epochs, after every epoch by default",
)
parser.add_argument(
    "--validate_at_start",
    action="store_true",
    default=False,
    help="whether to run evaluation before training starts",
)
parser.add_argument(
    "--save_freq",
    type=int,
    default=100,
    help="frequency of trajectory visualization during training",
)
parser.add_argument(
    "--eval_max_seq_len",
    type=int,
    default=1000,
    help="maximum length of evaluation videos",
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="if using the visibility mask",
)
parser.add_argument(
    "--fine_tune",
    action="store_true",
    default=False,
    help="if fine tune the model",
)
parser.add_argument(
    "--aug_wind_sample",
    action="store_true",
    default=False,
    help="if using the window sampling",
)
parser.add_argument(
    "--use_video_flip",
    action="store_true",
    default=False,
    help="if using the video flip",
)
parser.add_argument(
    "--fix_backbone",
    action="store_true",
    default=False,
    help="if fix the backbone",
)
parser.add_argument(
    "--tune_backbone",
    action="store_true",
    default=False,
    help="if tune the backbone",
)
parser.add_argument(
    "--tune_arap",
    action="store_true",
    default=False,
    help="if fix the backbone",
)
parser.add_argument(
    "--tune_per_scene",
    action="store_true",
    default=False,
    help="if tune one scene",
)
parser.add_argument(
    "--use_hier_encoder",
    action="store_true",
    default=False,
    help="if using the hierarchical encoder",
)
parser.add_argument(
    "--scales",
    type=int,
    nargs="+",
    default=[4, 2],
    help="scales for the CoTracker feature network",
)

# config for monocular depth estimator
parser.add_argument(
    "--mde_name", type=str, default="zoedepth_nk", help="name of the MDE model"
)
args = parser.parse_args()
args_dict = vars(args)

# -----------------------------------------------------------------------------

# merge the `args` to the `cfg`
cfg.merge_from_dict(args_dict)

cfg.ckpt_path=os.path.join(args.save_dir, args.model_name ,args.exp_name)

