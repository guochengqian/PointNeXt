"""
(Distributed) training script for scene segmentation
This file currently supports training and testing on S3DIS
If more than 1 GPU is provided, will launch multi processing distributed training by default
if you only wana using 1 GPU, set `CUDA_VISIBLE_DEVICES` accordingly
Author: Guocheng Qian @ 2022, guocheng.qian@kaust.edu.sa
"""
import __init__
import argparse, yaml, os, logging, numpy as np, csv, wandb
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg, get_scene_seg_features, get_class_weights
from openpoints.dataset.data_util import voxelize
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def write_to_csv(oa, macc, miou, ious, best_epoch, cfg, write_header=True, area=5):
    ious_table = [f'{item:.2f}' for item in ious]
    header = ['method', 'Area', 'OA', 'mACC', 'mIoU'] + cfg.classes + ['best_epoch', 'log_path', 'wandb link']
    data = [cfg.cfg_basename, str(area), f'{oa:.2f}', f'{macc:.2f}', f'{miou:.2f}'] + ious_table + [str(best_epoch), cfg.run_dir, wandb.run.get_url() if cfg.wandb.use_wandb else '-']
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()


def main(gpu, cfg):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0:
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir) if cfg.is_training else None
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    num_classes = val_loader.dataset.num_classes if hasattr(val_loader.dataset, 'num_classes') else None
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    logging.info(f"number of classes of the dataset: {num_classes}")
    cfg.classes = val_loader.dataset.classes if hasattr(val_loader.dataset, 'classes') else np.range(num_classes)
    cfg.cmap = np.array(val_loader.dataset.cmap) if hasattr(val_loader.dataset, 'cmap') else None
    validate_fn = validate if 'sphere' not in cfg.dataset.common.NAME.lower() else validate_sphere
    
    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler, pretrained_path=cfg.pretrained_path)
            val_miou = validate_fn(val_loader, model, cfg)
            logging.info(f'\nresume val miou is {val_miou}\n ')
        else:
            if cfg.mode == 'val':
                best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg, num_votes=1)
                with np.printoptions(precision=2, suppress=True):
                    logging.info(f'Best ckpt @E{best_epoch},  val_oa , val_macc, val_miou: {oa:.2f} {macc:.2f} {miou:.2f}, '
                                f'\niou per cls is: {ious}')
                return miou 
            elif cfg.mode == 'test':
                best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                miou, macc, oa, ious, accs, _ = test_entire_room(model, cfg.dataset.common.test_area, cfg)
                with np.printoptions(precision=2, suppress=True):
                    logging.info(f'Best ckpt @E{best_epoch},  test_oa , test_macc, test_miou: {oa:.2f} {macc:.2f} {miou:.2f}, '
                                f'\niou per cls is: {ious}')
                cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + '_test.csv')
                write_to_csv(oa, macc, miou, ious, best_epoch, cfg)
                return miou 
            else:
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                model.load_model_from_ckpt(cfg.pretrained_path, only_encoder=cfg.only_encoder)
    else:
        logging.info('Training from scratch')

    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")
    
    cfg.criterion.weight = None 
    if cfg.get('cls_weighed_loss', False):
        if hasattr(train_loader.dataset, 'num_per_class'):    
            cfg.criterion.weight = get_class_weights(train_loader.dataset.num_per_class, normalize=True)
        else: 
            logging.info('`num_per_class` attribute is not founded in dataset')
    criterion = build_criterion_from_cfg(cfg.criterion).cuda()

    # ===> start training
    val_miou, val_macc, val_oa, val_ious, val_accs = 0., 0., 0., [], []
    best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):  # some dataset sets the dataset length as a fixed steps.
            train_loader.dataset.epoch = epoch - 1
        train_loss, train_miou, train_macc, train_oa, _, _ = \
            train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, cfg)

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg)
            if val_miou > best_val:
                is_best = True
                best_val = val_miou
                macc_when_best = val_macc
                oa_when_best = val_oa
                ious_when_best = val_ious
                best_epoch = epoch
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Find a better ckpt @E{epoch}, val_miou {val_miou:.2f} val_macc {macc_when_best:.2f}, val_oa {oa_when_best:.2f}'
                        f'\nmious: {val_ious}')

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_miou {train_miou:.2f}, val_miou {val_miou:.2f}, best val miou {best_val:.2f}')
        if writer is not None:
            writer.add_scalar('best_val', best_val, epoch)
            writer.add_scalar('val_miou', val_miou, epoch)
            writer.add_scalar('macc_when_best', macc_when_best, epoch)
            writer.add_scalar('oa_when_best', oa_when_best, epoch)
            writer.add_scalar('val_macc', val_macc, epoch)
            writer.add_scalar('val_oa', val_oa, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_miou', train_miou, epoch)
            writer.add_scalar('train_macc', train_macc, epoch)
            writer.add_scalar('lr', lr, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )
            is_best = False
    # do not save file to wandb to save wandb space
    # if writer is not None:
    #     Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    # Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.logname}_ckpt_latest.pth'))

    # validate
    with np.printoptions(precision=2, suppress=True):
        logging.info(
            f'Best ckpt @E{best_epoch},  val_oa {oa_when_best:.2f}, val_macc {macc_when_best:.2f}, val_miou {best_val:.2f}, '
            f'\niou per cls is: {ious_when_best}')
    
    # test
    load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + f'_area5.csv')
    if 'sphere' in cfg.dataset.common.NAME.lower():
        test_miou, test_macc, test_oa, test_ious, test_accs = validate_sphere(model, val_loader, cfg)
    else:
        test_miou, test_macc, test_oa, test_ious, test_accs, _  = test_entire_room(model, cfg.dataset.common.test_area, cfg)
    with np.printoptions(precision=2, suppress=True):
        logging.info(f'Best ckpt @E{best_epoch},  test_oa {test_oa:.2f}, test_macc {test_macc:.2f}, test_miou {test_miou:.2f}, '
                    f'\niou per cls is: {test_ious}')
    if writer is not None:
        writer.add_scalar('test_miou', test_miou, epoch)
        writer.add_scalar('test_macc', test_macc, epoch)
        writer.add_scalar('test_oa', test_oa, epoch)
    write_to_csv(test_oa, test_macc, test_miou, test_ious, best_epoch, cfg, write_header=True)
    logging.info(f'save results in {cfg.csv_path}')
    if cfg.use_voting:
        load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
        set_random_seed(cfg.seed)
        val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg, num_votes=20,
                                                                     data_transform=data_transform)
        if writer is not None:
            writer.add_scalar('val_miou20', val_miou, cfg.epochs + 50)

        ious_table = [f'{item:.2f}' for item in val_ious]
        data = [cfg.cfg_basename, 'True', f'{val_oa:.2f}', f'{val_macc:.2f}', f'{val_miou:.2f}'] + ious_table + [
            str(best_epoch), cfg.run_dir]
        with open(cfg.csv_path, 'w', encoding='UT8') as f:
            writer = csv.writer(f)
            writer.writerow(data)
    return True


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        target = data['y'].squeeze(-1)
        # # debug
        # from openpoints.dataset import vis_points
        # vis_points(data['pos'].cpu().numpy()[0], labels=data['y'].cpu().numpy()[0])
        # vis_points(data['pos'].cpu().numpy()[0], data['x'][0, :3, :].transpose(1, 0))
        if len(data['x'].shape) > 2:
            data['x'] = data['x'].transpose(1, 2)
        data['x'] = get_scene_seg_features(cfg.model.in_channels, data['pos'], data['x'])

        logits = model(data)
        loss = criterion(logits, target) if 'mask' not in cfg.criterion.NAME.lower() \
            else criterion(logits, target, data['mask'])
        loss.backward()
        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # update confusion matrix
        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())

        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
    miou, macc, oa, ious, accs = cm.all_metrics()
    return loss_meter.avg, miou, macc, oa, ious, accs


@torch.no_grad()
def validate(model, val_loader, cfg, num_votes=1, data_transform=None):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y'].squeeze(-1)
        if len(data['x'].shape) > 2:
            data['x'] = data['x'].transpose(1, 2)
        data['x'] = get_scene_seg_features(cfg.model.in_channels, data['pos'], data['x'])

        logits = model(data)
        # if 'mask' not in cfg.criterion.NAME or cfg.get('use_maks', False):
        cm.update(logits.argmax(dim=1), target)
        # else:
        #     mask = data['mask'].bool()
        #     cm.update(logits.argmax(dim=1)[mask], target[ma])
    tp, union, count = cm.tp, cm.union, cm.count
    # print(f'before gathering, rank {cfg.rank} \n tp {tp} \n union {union} \n count {count}')
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
    # print(f'after gathering, rank {cfg.rank} \n union {union} \n count {count}')
    miou, macc, oa, ious, accs = get_mious(tp, union, count)
    return miou, macc, oa, ious, accs


@torch.no_grad()
def validate_sphere(model, val_loader, cfg, num_votes=1, data_transform=None):
    """
    validation for sphere sampled input points with mask.
    in this case, between different batches, there are overlapped points.
    thus, one point can be evaluated multiple times.
    In this validate_mask, we will avg the logits.
    """
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    all_logits, all_point_inds = [], []
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        if len(data['x'].shape) > 2:
            data['x'] = data['x'].transpose(1, 2)
        data['x'] = get_scene_seg_features(cfg.model.in_channels, data['pos'], data['x'])
        logits = model(data)
        all_logits.append(logits)
        all_point_inds.append(data['input_inds'])

    all_logits = torch.cat(all_logits, dim=0).transpose(1, 2).reshape(-1, cfg.num_classes)
    all_point_inds = torch.cat(all_point_inds, dim=0).flatten()

    if cfg.distributed:
        dist.all_reduce(all_logits), dist.all_reduce(all_point_inds)

    # subsampled points: 5323316 points.
    all_logits = scatter(all_logits, all_point_inds, dim=0, reduce='mean')

    # now, project the original points to the subsampled points
    # these two targets would be very similar but not the same
    # val_points_targets = all_targets[val_points_projections]
    # torch.allclose(val_points_labels, val_points_targets)
    all_logits = all_logits.argmax(dim=1)
    val_points_labels = torch.from_numpy(val_loader.dataset.clouds_points_labels[0]).squeeze(-1).to(all_logits.device)
    val_points_projections = torch.from_numpy(val_loader.dataset.projections[0]).to(all_logits.device).long()
    val_points_preds = all_logits[val_points_projections]

    del all_logits, all_point_inds
    torch.cuda.empty_cache()

    cm.update(val_points_preds, val_points_labels)
    miou, macc, oa, ious, accs = cm.all_metrics()
    return miou, macc, oa, ious, accs


@torch.no_grad()
def test_entire_room(model, area, cfg, global_cm=None, num_votes=1):
    """using a part of original point cloud as input to save memory.
    Args:
        model (_type_): _description_
        test_loader (_type_): _description_
        cfg (_type_): _description_
        global_cm (_type_, optional): _description_. Defaults to None.
        num_votes (int, optional): _description_. Defaults to 1.
    Returns:
        _type_: _description_
    """
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    global_cm =  ConfusionMatrix(num_classes=cfg.num_classes) if global_cm is None else global_cm
    set_random_seed(0)
    cfg.visualize = cfg.get('visualize', False)
    if cfg.visualize:
        from openpoints.dataset.vis3d import write_obj
        cfg.vis_dir = os.path.join(cfg.run_dir, 'visualization')
        os.makedirs(cfg.vis_dir, exist_ok=True)
        cfg.cmap = cfg.cmap.astype(np.float32) / 255.
        
    # data
    trans_split = 'val' if cfg.datatransforms.get('test', None) is None else 'test'
    transform =  build_transforms_from_cfg(trans_split, cfg.datatransforms)

    raw_root = os.path.join(cfg.dataset.common.data_root, 'raw')
    data_list = sorted(os.listdir(raw_root))
    data_list = [item[:-4] for item in data_list if 'Area_' in item]
    data_list = [item for item in data_list if 'Area_{}'.format(area) in item]

    voxel_size =  cfg.dataset.common.voxel_size
    for cloud_idx, item in enumerate(tqdm(data_list)):
        data_path = os.path.join(raw_root, item + '.npy')
        cdata = np.load(data_path).astype(np.float32)  # xyz, rgb, label, N*7
        coord_min = np.min(cdata[:, :3], 0)
        cdata[:, :3] -= coord_min
        label = torch.from_numpy(cdata[:, 6].astype(np.int).squeeze()).cuda(non_blocking=True)
        colors = np.clip(cdata[:, 3:6] / 255., 0, 1).astype(np.float32)

        all_logits, all_point_inds = [], []
        if voxel_size is not None:
            uniq_idx, count = voxelize(cdata[:, :3], voxel_size, mode=1)
            for i in range(count.max()):
                idx_select = np.cumsum(
                    np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = uniq_idx[idx_select]
                np.random.shuffle(idx_part)
                all_point_inds.append(idx_part)
                coord, feat = cdata[idx_part][:,0:3] - np.min(cdata[idx_part][:, :3], 0), cdata[idx_part][:, 3:6]

                data = {'pos': coord, 'x': feat}
                if transform is not None:
                    data = transform(data)
                if 'heights' in data.keys():
                    data['x'] = torch.cat((data['x'], data['heights']), dim=1)
                else:
                    data['x'] = torch.cat((data['x'], torch.from_numpy(
                        coord[:, 3-cfg.dataset.common.get('n_shifted', 1):3].astype(np.float32))), dim=-1)

                if not cfg.dataset.common.get('variable', False):
                    data['x'] = data['x'].transpose(1, 0).unsqueeze(0)
                    data['pos'] = data['pos'].unsqueeze(0)
                else:
                    data['o'] = torch.IntTensor([len(coord)])

                keys = data.keys() if callable(data.keys) else data.keys
                for key in keys:
                    data[key] = data[key].cuda(non_blocking=True)
                data['x'] = get_scene_seg_features(cfg.model.encoder_args.in_channels, data['pos'], data['x'])

                logits = model(data)
                all_logits.append(logits)
                """visualization in debug mode 
                from openpoints.dataset.vis3d import vis_points, vis_multi_points
                # vis_points(cdata[:, :3], cdata[:, 3:6]/255.)
                # vis_multi_points([cdata[:, :3], coord], [cdata[:, 3:6].astype(np.uint8), feat.astype(np.uint8)])
                vis_multi_points([cdata[:, :3], coord], labels=[label.cpu().numpy(), logits.argmax(dim=1).squeeze().cpu().numpy()])
                """
            all_logits = torch.cat(all_logits, dim=0)
            if not cfg.dataset.common.get('variable', False):
                all_logits = all_logits.transpose(1, 2).reshape(-1, cfg.num_classes)
            all_point_inds = torch.from_numpy(np.hstack(all_point_inds)).cuda(non_blocking=True)

            if cfg.distributed:
                dist.all_reduce(all_logits), dist.all_reduce(all_point_inds)

            # project voxel subsampled to original set
            all_logits = scatter(all_logits, all_point_inds, dim=0, reduce='mean')
            all_point_inds = scatter(all_point_inds, all_point_inds, dim=0, reduce='mean')

            cm.update(all_logits.argmax(dim=1), label)
            global_cm.update(all_logits.argmax(dim=1), label)
            
            if cfg.visualize:
                gt = label.cpu().numpy().squeeze()
                pred = all_logits.argmax(dim=1).cpu().numpy().squeeze()
                gt = cfg.cmap[gt, :]
                pred = cfg.cmap[pred, :]
                # output pred labels
                write_obj(cdata[:, :3], colors,
                          os.path.join(cfg.vis_dir, f'input-Area{area}-{cloud_idx}.obj'))
                # output ground truth labels
                write_obj(cdata[:, :3], gt,
                          os.path.join(cfg.vis_dir, f'gt-Area{area}-{cloud_idx}.obj'))
                # output pred labels
                write_obj(cdata[:, :3], pred,
                          os.path.join(cfg.vis_dir, f'pred-Area{area}-{cloud_idx}.obj'))
    tp, union, count = cm.tp, cm.union, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
    miou, macc, oa, ious, accs = get_mious(tp, union, count)
    return miou, macc, oa, ious, accs, global_cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)    # overwrite the default arguments in yml 

    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis, modelnet40_cls
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]   # cfg_basename, \eg pointnext-xl 
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            tags.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)

    cfg.is_training = cfg.mode in ['train', 'training', 'finetune', 'finetuning']
    if cfg.mode == 'train':
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    else:  # resume from the existing ckpt and reuse the folder.
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = ['resume']
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # wandb config
    cfg.wandb.name = cfg.run_name

    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg))
    else:
        main(0, cfg)
