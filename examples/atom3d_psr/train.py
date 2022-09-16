"""
Distributed training script for scene segmentation with S3DIS dataset
"""
import os
import sys
import time
import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import logging
from tqdm import tqdm

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from openpoints.utils import AverageMeter, set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, \
    cal_model_parm_nums

from openpoints.dataset import build_dataloader_from_cfg
from openpoints.models.build import build_model_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg


def get_features_by_keys(input_features_dim, pc, node_feature):
    if input_features_dim > 3:
        return node_feature
    else:
        return None


def compute_correlations(results):
    per_target = []
    for key, val in results.groupby(['target']):
        # Ignore target with 2 decoys only since the correlations are
        # not really meaningful.
        if val.shape[0] < 3:
            continue
        true = val['true'].astype(float)
        pred = val['pred'].astype(float)
        pearson = true.corr(pred, method='pearson')
        kendall = true.corr(pred, method='kendall')
        spearman = true.corr(pred, method='spearman')
        per_target.append((key, pearson, kendall, spearman))
    per_target = pd.DataFrame(
        data=per_target,
        columns=['target', 'pearson', 'kendall', 'spearman'])

    res = {}
    all_true = results['true'].astype(float)
    all_pred = results['pred'].astype(float)
    res['all_pearson'] = all_true.corr(all_pred, method='pearson')
    res['all_kendall'] = all_true.corr(all_pred, method='kendall')
    res['all_spearman'] = all_true.corr(all_pred, method='spearman')

    res['per_target_pearson'] = per_target['pearson'].mean()
    res['per_target_kendall'] = per_target['kendall'].mean()
    res['per_target_spearman'] = per_target['spearman'].mean()

    print(
        '\nCorrelations (Pearson, Kendall, Spearman)\n'
        '    per-target: ({:.3f}, {:.3f}, {:.3f})\n'
        '    global    : ({:.3f}, {:.3f}, {:.3f})'.format(
        float(res["per_target_pearson"]),
        float(res["per_target_kendall"]),
        float(res["per_target_spearman"]),
        float(res["all_pearson"]),
        float(res["all_kendall"]),
        float(res["all_spearman"])))
    return res


def run_net(config, writer, profile=False):
    # TODO: have to change to torch geometric data loader.
    train_loader = build_dataloader_from_cfg(config.batch_size,
                                             config.dataset, 
                                             dataloader_cfg=config.dataloader,
                                             datatransforms_cfg=config.datatransforms, 
                                             split='train'
                                             )
    val_loader = build_dataloader_from_cfg(config.batch_size,
                                           config.dataset, 
                                           dataloader_cfg=config.dataloader,
                                           datatransforms_cfg=config.datatransforms, 
                                           split='val'
                                           )
    test_loader = build_dataloader_from_cfg(config.batch_size,
                                            config.dataset, 
                                            dataloader_cfg=config.dataloader,
                                            datatransforms_cfg=config.datatransforms, 
                                            split='test'
                                            )
    
    model = build_model_from_cfg(config.model).to(config.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / (1e6)))

    criterion = torch.nn.MSELoss()  # regression problem

    if profile:
        model.eval()
        total_time = 0.
        # points, mask, features, points_labels, cloud_label, input_inds = iter(val_loader).next()
        # points = points.cuda(non_blocking=True)
        # features = features.cuda(non_blocking=True)
        # print(points.shape, features.shape)
        B, N, C = 16, 16384, config.model.in_channels
        points = torch.randn(B, N, 3).cuda()
        features = torch.randn(B, C, N).cuda()
        mask = None

        # from thop import profile as thop_profile
        # macs, params = thop_profile(model, inputs=(points, features))
        # macs = macs / 1e6
        # params = params / 1e6
        # logging.info(f'mac: {macs} \nparams: {params}')

        n_runs = 300
        with torch.no_grad():
            for idx in range(n_runs):
                start_time = time.time()
                model(points, features)
                torch.cuda.synchronize()
                time_taken = time.time() - start_time
                total_time += time_taken
        print(f'inference time: {total_time / float(n_runs)}')
        return False

    # optionally resume from a checkpoint
    if config.resume:
        resume_checkpoint(config, model, optimizer, scheduler)
    else:
        if config.load_path is not None:
            if config.mode=='finetune':
                model.load_model_from_ckpt(config.load_path)
            else:
                load_checkpoint(model, config.load_path)
                rmse, corrs, test_df = validate(test_loader, model, criterion, config)
                logging.info('Test RMSE: {:.7f}, Per-target Spearman R: {:.7f}, Global Spearman R: {:.7f}'.format(
                    rmse, corrs['per_target_spearman'], corrs['all_spearman']))
                return 

    if config.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    model = DistributedDataParallel(model, device_ids=[config.rank], output_device=config.rank)

    optimizer = build_optimizer_from_cfg(model, lr=config.lr, **config.optimizer)
    scheduler = build_scheduler_from_cfg(config, optimizer)
    # Then load optimizer and scheduler. TODO. 
        
    # ===> start training
    loss = 0.
    val_loss = 0.
    best_val_loss = np.inf
    is_best = False
    mean_rs= 0.
    global_rs = 0.
    for epoch in range(config.start_epoch, config.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        train_loader.dataset.epoch = epoch - 1
        tic = time.time()
        loss = train(epoch, train_loader, model, criterion, optimizer, scheduler, config)
        if epoch % config.val_freq == 0:
            val_loss, corrs, val_df = validate(val_loader, model, criterion, config)
            mean_rs, global_rs = corrs['per_target_spearman'], corrs['all_spearman']
            if val_loss < best_val_loss:
                is_best = True
                best_val_loss = val_loss
                best_corrs = corrs

        logging.info('epoch {}, total time {:.2f}, lr {:.5f}, '
                    'best val loss {:3f}'.format(epoch,
                                                 (time.time() - tic),
                                                 optimizer.param_groups[0]['lr'],
                                                 best_val_loss))
        if dist.get_rank() == 0:
            # save model
            save_checkpoint(config, model, epoch, optimizer, scheduler,
                            additioanl_dict={'corrs': best_corrs},
                            is_best=is_best)
            is_best = False

        if writer is not None:
            # tensorboard logger
            writer.add_scalar('best_val_loss', best_val_loss, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_mean_rs', mean_rs, epoch)
            writer.add_scalar('val_global_rs', global_rs, epoch)
            writer.add_scalar('train_loss', loss, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if config.sched_on_epoch:
            scheduler.step(epoch)

        if epoch % 20 == 0:
            # if writer is not None:
            #     Wandb.add_file(os.path.join(config.ckpt_dir, f'{config.run_name}_ckpt_best.pth'))
            # Wandb.add_file(os.path.join(config.ckpt_dir, f'{config.logname}_ckpt_latest.pth'))
            load_checkpoint(model, pretrained_path=os.path.join(config.ckpt_dir, f'{config.run_name}_ckpt_best.pth'))
            set_random_seed(config.seed)
            rmse, corrs, test_df = validate(test_loader, model, criterion, config)
            logging.info('Test RMSE: {:.7f}, Per-target Spearman R: {:.7f}, Global Spearman R: {:.7f}'.format(
                rmse, corrs['per_target_spearman'], corrs['all_spearman']))
            if writer is not None:
                # tensorboard logger
                writer.add_scalar('RS', corrs['per_target_spearman'], epoch)
                writer.add_scalar('global_RS', corrs['all_spearman'], epoch)
    # test_df.to_pickle(os.path.join(args.output_dir, 'test_results.pkl'))


def train(epoch, train_loader, model, criterion, optimizer, scheduler, config):
    """
    One epoch training
    """
    model.train()
    optimizer.zero_grad()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    end = time.time()

    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        points = data['pos'].cuda(non_blocking=True)
        node_features = data['features'].cuda(non_blocking=True)
        label = data['label'].cuda(non_blocking=True)
        features = get_features_by_keys(config.model.in_chans, points, node_features)

        # # debug
        # from openpoints.dataset import vis_points
        # vis_points(data['pos'][0])

        pred = model(points, features)
        loss = criterion(pred, label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)

        if ((idx + 1) % config.step_per_update == 0) or (idx + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
            if not config.sched_on_epoch:
                scheduler.step()

        # update meters
        # TODO: Later, support batch
        loss_meter.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % config.print_freq == 0:
            logging.info(f'Train: [{epoch}/{config.epochs}][{idx}/{len(train_loader)}]\t'
                        f'T {batch_time.val(): .3f} ({batch_time.avg():.3f})\t'
                        f'DT {data_time.val():.3f} ({data_time.avg():.3f})\t'
                        f'loss {loss_meter.val():.3f} ({loss_meter.avg():.3f})')
    return loss_meter.avg()


@torch.no_grad()
def validate(test_loader, model, criterion, config):
    """
    One epoch validating
    """

    losses = AverageMeter()
    model.eval()

    test_loader.dataset.epoch = 0

    targets = []
    decoys = []
    y_true = []
    y_pred = []

    for idx, data in enumerate(tqdm(test_loader)):
        # augment for voting
        points = data['pos'].cuda(non_blocking=True)
        label = data['label'].cuda(non_blocking=True)
        node_features = data['features'].cuda(non_blocking=True)
        features = get_features_by_keys(config.model.in_chans, points, node_features)

        pred = model(points, features)
        loss = criterion(pred, label)
        losses.update(loss.item())

        targets.extend(data['target'])
        decoys.extend(data['decoy'])
        y_true.extend(label.tolist())
        y_pred.extend(pred.tolist())

    targets = collect_results_gpu(targets, test_loader.dataset.__len__())
    decoys = collect_results_gpu(decoys, test_loader.dataset.__len__())
    y_true = collect_results_gpu(y_true, test_loader.dataset.__len__())
    y_pred = collect_results_gpu(y_pred, test_loader.dataset.__len__())
    results_df = pd.DataFrame(
        np.array([targets, decoys, y_true, y_pred]).T,
        columns=['target', 'decoy', 'true', 'pred'],
    )

    corrs = compute_correlations(results_df)
    return np.sqrt(losses.avg()), corrs, results_df


def collect_results_gpu(result_part, size):
    import pickle
    rank, world_size = dist.get_rank(), dist.get_world_size()
    # dump result part to tensor with pickle
    if world_size == 1:
        return result_part
    else:
        part_tensor = torch.tensor(
            bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
        # gather all result part tensor shape
        shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
        shape_list = [shape_tensor.clone() for _ in range(world_size)]
        dist.all_gather(shape_list, shape_tensor)
        # padding result part tensor to max length
        shape_max = torch.tensor(shape_list).max()
        part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
        part_send[:shape_tensor[0]] = part_tensor
        part_recv_list = [
            part_tensor.new_zeros(shape_max) for _ in range(world_size)
        ]
        # gather all result part
        dist.all_gather(part_recv_list, part_send)

        if rank == 0:
            part_list = []
            for recv, shape in zip(part_recv_list, shape_list):
                part_list.append(
                    pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
            # sort the results
            ordered_results = []
            for res in zip(*part_list):
                ordered_results.extend(list(res))
            # the dataloader may pad some samples
            original_size = len(ordered_results)
            ordered_results = ordered_results[:size]
            print(f'total length of preditions of {original_size} is reduced to {size}')
            return ordered_results

