"""
    This file include some utils for training
"""

import os
import torch
from . import cluster

cluster_x = cluster.cluster_x

def ckpt_from_folder(folder_path):
    """
    Get the checkpoint path from the folder
    """

    folder_ckpts = [
        f
        for f in os.listdir(folder_path)
        if not os.path.isdir(f) and f.endswith(".pth") and not "final" in f
    ]

    return [] if len(folder_ckpts) == 0 else sorted(folder_ckpts)[-1]


def load_ckpt(args,
              model, 
              optimizer, 
              scheduler,
              logging):
    """
        load the checkpoints from two modes:
        1. from the folder
        2. from the specific file
        3. do not load any checkpoint
    """

    if args.restore_ckpt == "":
        # load the checkpoint from the folder
        ckpt_path = ckpt_from_folder(args.ckpt_path)
        if ckpt_path == []:
            return model, optimizer, scheduler, 0
        ckpt_path = os.path.join(args.ckpt_path, ckpt_path)
    else:
        # load the checkpoint from the specific file
        ckpt_path = args.restore_ckpt

    # load the checkpoint
    ckpt = torch.load(ckpt_path)
    model_paras = model.state_dict()
    if "model" in ckpt:
        if args.fix_backbone == True:
            paras_dict = {k: v for k,v in ckpt["model"].items() if (k in model.state_dict()) and ("fnet" in k)}
            paras_dict.update({k: v for k,v in ckpt["model"].items() if ("embedConv" in k and k in model.state_dict().keys())})
        else:
            paras_dict = {k: v for k,v in ckpt["model"].items() if k in model.state_dict()}
        model_paras.update(paras_dict)
    else:
        if args.fix_backbone == True:
            paras_dict = {k: v for k,v in ckpt.items() if (k in model.state_dict()) and ("fnet" in k)}
            paras_dict.update({k: v for k,v in ckpt.items() if ("embedConv" in k and k in model.state_dict().keys())})
        else:
            paras_dict = {k: v for k,v in ckpt.items() if k in model.state_dict()}
        model_paras.update(paras_dict)
    model.load_state_dict(model_paras)
    
    # if fine tune the ckpt from the pretrained model 
    if args.fine_tune == True:
        logging.info("Fine tune the model")
        total_steps = 0
    else:
        if "optimizer" in ckpt:
            logging.info("Load optimizer")
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            logging.info("Load scheduler")
            scheduler.load_state_dict(ckpt["scheduler"])
        if "total_steps" in ckpt:
            total_steps = ckpt["total_steps"]
            logging.info(f"Load total_steps {total_steps}")
        
    return model, optimizer, scheduler, total_steps

def load_hier_ckpt(args,
              model, 
              optimizer, 
              scheduler,
              logging):
    """
        load the checkpoints from two modes:
        1. from the folder
        2. from the specific file
        3. do not load any checkpoint
    """

    if args.restore_ckpt == "":
        # load the checkpoint from the folder
        ckpt_path = ckpt_from_folder(args.ckpt_path)
        if ckpt_path == []:
            return model, optimizer, scheduler, 0
        ckpt_path = os.path.join(args.ckpt_path, ckpt_path)
    else:
        # load the checkpoint from the specific file
        ckpt_path = args.restore_ckpt

    # load the checkpoint
    ckpt = torch.load(ckpt_path)
    model_paras = model.state_dict()

    paras_dict = {k: v for k,v in ckpt.items() if k in model.state_dict()}
    model_paras.update(paras_dict)
    model.load_state_dict(model_paras)
    
    # if fine tune the ckpt from the pretrained model 
    if args.fine_tune == True:
        logging.info("Fine tune the model")
        total_steps = 0
    else:
        if "optimizer" in ckpt:
            logging.info("Load optimizer")
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            logging.info("Load scheduler")
            scheduler.load_state_dict(ckpt["scheduler"])
        if "total_steps" in ckpt:
            total_steps = ckpt["total_steps"]
            logging.info(f"Load total_steps {total_steps}")
        
    return model, optimizer, scheduler, total_steps


