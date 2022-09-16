import torch
import torch.nn as nn
import time
from openpoints.utils import AverageMeter, resume_model, resume_optimizer, load_checkpoint, save_checkpoint, reduce_tensor, \
    gather_tensor, cal_model_parm_nums

import numpy as np

from openpoints.dataset import build_dataloader_from_cfg, build_transforms_from_cfg
from openpoints.models.build import build_model_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
import logging


def get_features_by_keys(input_features_dim, pc, node_feature):
    if input_features_dim > 3:
        return node_feature
    else:
        return None


def run_net(config, writer):
    train_dataloader = build_dataloader_from_cfg(config.batch_size,
                                             config.dataset, 
                                             dataloader_cfg=config.dataloader,
                                             datatransforms_cfg=config.datatransforms, 
                                             split='train'
                                             )

    # build model
    base_model = build_model_from_cfg(config.model)
    model_size = cal_model_parm_nums(base_model)
    logging.info(base_model)
    logging.info('Number of params: %.4f M' % (model_size / (1e6)))

    # resume pretrained_path
    if config.resume:
        config.start_epoch, best_metric = resume_model(base_model, config)
    else:
        if config.pretrained_path is not None:
            load_checkpoint(base_model, config.pretrained_path)
        else:
            logging.info('Training from scratch')

    if config.use_gpu:
        base_model.cuda()
    if config.distributed:
        # Sync BN
        if config.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            logging.info('Using Synchronized BatchNorm ...')
        base_model = nn.parallel.DistributedDataParallel(base_model,device_ids=[config.rank])
        logging.info('Using Distributed Data parallel ...')
    else:
        # logging.info('Using Data parallel ...' )
        # base_model = nn.DataParallel(base_model).cuda()
        raise NotImplementedError

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(base_model, lr=config.lr, **config.optimizer)
    scheduler = build_scheduler_from_cfg(config, optimizer)

    if config.resume:
        resume_optimizer(optimizer, config)

    base_model.zero_grad()
    for epoch in range(config.start_epoch, config.epochs + 1):
        if config.distributed:
            # In distributed mode, calling the set_epoch() method at the beginning of each epoch 
            # before creating the DataLoader iterator is necessary to make shuffling work properly 
            # across multiple epochs. 
            # Otherwise, the same ordering will be always used. 
            train_dataloader.sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        
        for idx, data in enumerate(train_dataloader): 
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)
            points = data['pos'].cuda()
            node_features = data['features'].cuda(non_blocking=True)
            features = get_features_by_keys(config.model.in_chans, points, node_features)
            
            loss, pred = base_model(points, features)
            
            loss.backward()
            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                optimizer.zero_grad()

            # if config.distributed:
            #     loss = reduce_tensor(loss)
            # else:
            #     raise NotImplementedError
            losses.update(loss.item())

            # if config.distributed:
            #     torch.cuda.synchronize()

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # if writer is not None:
            #     writer.add_scalar('train_loss', loss.item(), n_itr)
                
            if idx % config.print_freq == 0:
                logging.info(
                    '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %s lr = %.6f' %
                    (epoch, config.epochs, idx + 1, n_batches, batch_time.val(), data_time.val(),
                     losses.val(), optimizer.param_groups[0]['lr']))
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if writer is not None:
            writer.add_scalar('epoch_loss', losses.avg(), epoch)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
            
        logging.info('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
                     (epoch, epoch_end_time - epoch_start_time, losses.avg()))
        save_checkpoint(config, base_model, optimizer=None, scheduler=None, epoch=epoch, 
                        post_fix=f'_E{epoch}'
                        )
    if writer is not None:
        writer.close()

