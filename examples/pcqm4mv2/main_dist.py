"""
Distributed training script for pcqm4mv2
https://ogb.stanford.edu/docs/lsc/pcqm4mv2/
"""
import argparse
import os
import sys
import time
import yaml
import numpy as np
import logging
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from openpoints.utils import AverageMeter, set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    generate_exp_directory, resume_exp_directory, cal_model_parm_nums, Wandb, cfg

from openpoints.dataset import build_dataloader_from_cfg, build_transforms_from_cfg
from openpoints.models.build import build_model_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg


def main(config, profile=False):
    data_transform = build_transforms_from_cfg(config.datatransforms.train, config.datatransforms.kwargs)
    train_loader, val_loader, = build_dataloader_from_cfg(config.batch_size, config.dataset,
                                                          config.dataloader,
                                                        #   config.datatransforms,
                                                          split='train',
                                                          ), \
                                build_dataloader_from_cfg(config.batch_size, config.dataset,
                                                          config.dataloader,
                                                        #   config.datatransforms,
                                                          split='val')

    # n_data = len(train_loader.dataset)
    # logger.info(f"length of training dataset: {n_data}")
    # n_data = len(val_loader.dataset)
    # logger.info(f"length of validation dataset: {n_data}")
    if profile:
        cfg.model.norm_args.norm=None
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logger.info('Number of params: %.4f M' % (model_size / (1e6)))


    # criterion = MaskedCrossEntropy().cuda()

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

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    model = DistributedDataParallel(model, device_ids=[cfg.rank], output_device=cfg.rank)

    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    runing_vote_logits = [np.zeros((cfg.num_classes, l.shape[0]), dtype=np.float32) for l in
                          val_loader.dataset.sub_clouds_points_labels]
    # optionally resume from a checkpoint
    if cfg.mode == 'resume':
        resume_checkpoint(cfg, model, optimizer, scheduler)
        if 'train' in cfg.mode:
            val_miou = validate('resume', val_loader, model, criterion, runing_vote_logits, config, num_votes=2)
            logger.info(f'\nresume val mIoU is {val_miou}\n ')
        else:
            val_miou_20 = validate('Test', val_loader, model, criterion, runing_vote_logits, config, num_votes=20,
                                   data_transform=data_transform)
            logger.info(f'\nval mIoU is {val_miou_20}\n ')
            return val_miou_20

    # ===> start training
    val_miou = 0.
    best_val = 0.
    loss = 0.
    is_best = False
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        train_loader.dataset.epoch = epoch - 1
        tic = time.time()
        loss = train(epoch, train_loader, model, criterion, optimizer, scheduler, data_transform, config)
        if epoch % cfg.val_freq == 0:
            val_miou = validate(0, val_loader, model, criterion, runing_vote_logits, config, num_votes=1,
                                data_transform=data_transform)
            if val_miou > best_val:
                is_best = True
                best_val = val_miou

        logger.info('epoch {}, total time {:.2f}, lr {:.5f}, '
                    'best val mIoU {:3f}'.format(epoch,
                                                 (time.time() - tic),
                                                 optimizer.param_groups[0]['lr'],
                                                 best_val))
        if dist.get_rank() == 0:
            # save model
            save_checkpoint(config, model, epoch, optimizer, scheduler, is_best=is_best)
            is_best = False

        if writer is not None:
            # tensorboard logger
            writer.add_scalar('best_val_miou', best_val, epoch)
            writer.add_scalar('val_miou', val_miou, epoch)
            writer.add_scalar('ins_loss', loss, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)

    # if writer is not None:
    #     Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
        # Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.logname}_ckpt_latest.pth'))
    load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    set_random_seed(cfg.seed)
    best_miou_20 = validate('Best', val_loader, model, criterion, runing_vote_logits, config, num_votes=20,
                            data_transform=data_transform)
    if writer is not None:
        writer.add_scalar('val_miou20', best_miou_20, cfg.epochs + 50)


def train(epoch, train_loader, model, criterion, optimizer, scheduler, data_transform, config):
    """
    One epoch training
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    end = time.time()

    for idx, (data, mask, points_labels, _, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        data['pos'] = data['pos'].cuda(non_blocking=True)
        data['colors'] = data['colors'].cuda(non_blocking=True)
        data['height'] = data['height'].cuda(non_blocking=True)

        # # debug
        # import copy
        # ori_data = copy.deepcopy(data)
        # data = data_transform(data) # TODO: here, vis data before and after translation
        # from openpoints.dataset import vis_multi_points
        # vis_multi_points((ori_data['pos'].cpu().numpy()[0], data['pos'].cpu().numpy()[0]),
        #                  labels=(points_labels.cpu().numpy()[0], points_labels.cpu().numpy()[0]))

        data = data_transform(data)

        features = get_features_by_keys(config.model.in_channels, data['pos'],
                                          data['colors'].transpose(2, 1), data['height'].transpose(2, 1))

        # del data
        # bsz = points.size(0)

        mask = mask.cuda(non_blocking=True)
        points_labels = points_labels.cuda(non_blocking=True)


        pred = model(data['pos'], features)
        loss = criterion(pred, points_labels, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        if not cfg.sched_on_epoch:
            scheduler.step()

        # update meters
        # TODO: Later, support batch
        loss_meter.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % cfg.print_freq == 0:
            logger.info(f'Train: [{epoch}/{cfg.epochs}][{idx}/{len(train_loader)}]\t'
                        f'T {batch_time.val(): .3f} ({batch_time.avg():.3f})\t'
                        f'DT {data_time.val():.3f} ({data_time.avg():.3f})\t'
                        f'loss {loss_meter.val():.3f} ({loss_meter.avg():.3f})')
    return loss_meter.avg()


def validate(epoch, test_loader, model, criterion, runing_vote_logits, config,
             num_votes=1, data_transform=None):
    """
    One epoch validating
    """
    vote_logits_sum = [np.zeros((cfg.num_classes, l.shape[0]), dtype=np.float32) for l in
                       test_loader.dataset.sub_clouds_points_labels]
    vote_counts = [np.zeros((1, l.shape[0]), dtype=np.float32) + 1e-6 for l in
                   test_loader.dataset.sub_clouds_points_labels]
    vote_logits = [np.zeros((cfg.num_classes, l.shape[0]), dtype=np.float32) for l in
                   test_loader.dataset.sub_clouds_points_labels]
    validation_proj = test_loader.dataset.projections
    validation_labels = test_loader.dataset.clouds_points_labels
    test_smooth = 0.95

    val_proportions = np.zeros(cfg.num_classes, dtype=np.float32)
    for label_value in range(cfg.num_classes):
        val_proportions[label_value] = np.sum(
            [np.sum(labels == label_value) for labels in test_loader.dataset.clouds_points_labels])

    losses = AverageMeter()

    model.eval()
    with torch.no_grad():
        for v in range(num_votes):
            test_loader.dataset.epoch = (0 + v) if isinstance(epoch, str) else (epoch + v) % 20
            predictions = []
            targets = []

            for idx, (data, mask, points_labels, cloud_label, input_inds) in enumerate(tqdm(test_loader)):
                # augment for voting
                data['pos'] = data['pos'].cuda(non_blocking=True)
                data['colors'] = data['colors'].cuda(non_blocking=True)
                data['height'] = data['height'].cuda(non_blocking=True)
                if v > 0:
                    data = data_transform(data)
                features = get_features_by_keys(config.model.in_channels, data['pos'],
                                          data['colors'].transpose(2, 1), data['height'].transpose(2, 1))
                # forward
                mask = mask.cuda(non_blocking=True)
                points_labels = points_labels.cuda(non_blocking=True)
                cloud_label = cloud_label.cuda(non_blocking=True)
                input_inds = input_inds.cuda(non_blocking=True)

                pred = model(data['pos'], features)

                loss = criterion(pred, points_labels, mask)
                losses.update(loss.item())
                # losses.update(loss.item(), points.size(0))

                # collect
                bsz = data['pos'].shape[0]
                for ib in range(bsz):
                    mask_i = mask[ib].cpu().numpy().astype(bool)
                    logits = pred[ib].cpu().numpy()[:, mask_i]
                    inds = input_inds[ib].cpu().numpy()[mask_i]
                    c_i = cloud_label[ib].item()
                    vote_logits_sum[c_i][:, inds] = vote_logits_sum[c_i][:, inds] + logits
                    vote_counts[c_i][:, inds] += 1
                    vote_logits[c_i] = vote_logits_sum[c_i] / vote_counts[c_i]
                    runing_vote_logits[c_i][:, inds] = test_smooth * runing_vote_logits[c_i][:, inds] + \
                                                       (1 - test_smooth) * logits
                    predictions += [logits]
                    targets += [test_loader.dataset.sub_clouds_points_labels[c_i][inds]]

            predictions = collect_results_gpu(predictions, test_loader.dataset.__len__())
            targets = collect_results_gpu(targets, test_loader.dataset.__len__())

            mIoU = torch.tensor(0., device=torch.device('cuda'), dtype=torch.float64)
            if dist.get_rank() == 0:
                pIoUs, pmIoU = s3dis_part_metrics(cfg.num_classes, predictions, targets, val_proportions)
                runsubIoUs, runsubmIoU = sub_s3dis_metrics(cfg.num_classes, runing_vote_logits,
                                                           test_loader.dataset.sub_clouds_points_labels,
                                                           val_proportions)
                subIoUs, submIoU = sub_s3dis_metrics(cfg.num_classes, vote_logits,
                                                     test_loader.dataset.sub_clouds_points_labels, val_proportions)
                IoUs, mIoU = s3dis_metrics(cfg.num_classes, vote_logits, validation_proj, validation_labels)

                mIoU = torch.as_tensor(mIoU, device=torch.device('cuda'))
                logger.info(f'E{epoch} V{v} * part_mIoU {pmIoU:.3%}')
                logger.info(f'E{epoch} V{v}  * part_msIoU {pIoUs}')

                logger.info(f'E{epoch} V{v} * running sub_mIoU {runsubmIoU:.3%}')
                logger.info(f'E{epoch} V{v}  * running sub_msIoU {runsubIoUs}')

                logger.info(f'E{epoch} V{v} * sub_mIoU {submIoU:.3%}')
                logger.info(f'E{epoch} V{v}  * sub_msIoU {subIoUs}')

                logger.info(f'E{epoch} V{v} * mIoU {mIoU:.3%}')
                logger.info(f'E{epoch} V{v}  * msIoU {IoUs}')
            dist.broadcast(mIoU, 0)
    return mIoU


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
        cfg_path = os.path.join(cfg.run_dir, "cfg.json")
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
    main(cfg, profile=args.profile)
