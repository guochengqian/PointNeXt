"""
Distributed training script for scene segmentation with S3DIS dataset
"""
import argparse
import os
import sys
import time
import yaml
import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

import logging
from tqdm import tqdm

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from openpoints.utils import set_random_seed, setup_logger_dist, Wandb, cfg, generate_exp_directory, resume_exp_directory
from examples.atom3d_psr.train import run_net as train
from examples.atom3d_psr.pretrain import run_net as pretrain


if __name__ == "__main__":
    parser = argparse.ArgumentParser('S3DIS scene segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    cfg.local_rank = int(os.environ['LOCAL_RANK'])

    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cfg.rank = dist.get_rank()
    torch.backends.cudnn.enabled = True
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)

    # LR rule
    cfg.total_bs = cfg.batch_size * dist.get_world_size()
    cfg.lr = cfg.lr * dist.get_world_size()

    # logger
    if not cfg.mode == 'resume':
        cfg.exp_tag = args.cfg.split('.')[-2].split('/')[-1]
        tags = [
            args.cfg.split('.')[-2].split('/')[-2],
            cfg.mode,
            args.cfg.split('.')[-2].split('/')[-1],  # cfg file
            f'ngpus{dist.get_world_size()}',
        ]

        for i, opt in enumerate(opts):
            if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
                tags.append(opt)
        generate_exp_directory(cfg, tags, additional_id=os.environ['MASTER_PORT'])
        cfg.wandb.tags = tags
    else:  # resume from the existing ckpt and reuse the folder.
        resume_exp_directory(cfg, cfg.pretrained_path)
        cfg.wandb.tags = ['resume']
    logger = setup_logger_dist(cfg.log_path, dist.get_rank(), name="s3dis")  # stdout master only!
    os.environ["JOB_LOG_DIR"] = cfg.log_dir

    # wandb and tensorboard
    if dist.get_rank() == 0:
        cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
        with open(cfg_path, 'w') as f:
            yaml.dump(cfg, f, indent=2)
            os.system('cp %s %s' % (args.cfg, cfg.run_dir))
        cfg.cfg_path = cfg_path

        # wandb config
        cfg.wandb.name = cfg.run_name
        Wandb.launch(cfg, cfg.wandb.use_wandb)

        # tensorboard
        writer = SummaryWriter(log_dir=cfg.run_dir)
    else:
        writer = None

    logger.info(cfg)
    
    if cfg.mode == 'test':
        # make sure config.load_path points to the pretrained_model
        train(cfg, writer)
    elif cfg.mode == 'train' or cfg.mode == 'finetune':
        train(cfg, writer)
    elif cfg.mode == 'pretrain':
        pretrain(cfg, writer)
