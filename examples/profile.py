"""
Author: PointNeXt
for profiling the parameters, flops, speed of a model

usage example: 

1. profile pointnext-s on scanobjectnn using 128 * 1024 points as input
CUDA_VISIBLE_DEVICES=1 python examples/profile.py --cfg cfgs/scanobjectnn/pointnext-s.yaml batch_size=128 num_points=1024 timing=True

2. profile all models for scanobjectnn classification using 128 * 1024 points as input
CUDA_VISIBLE_DEVICES=1 python examples/profile.py --cfg cfgs/scanobjectnn batch_size=128 num_points=1024 timing=True
"""
import os, sys, argparse, time, warnings
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from openpoints.utils import EasyConfig, cal_model_parm_nums 
from openpoints.models import build_model_from_cfg
from deepspeed.profiling.flops_profiler import get_model_profile


def profile_model(model, cfg):
    model.eval()
    # for classification, num_points is 128 * 1024
    # for s3dis, num_points 16 * 15000 
    B, N, C = 1, cfg.num_points, cfg.model.in_channels
    if cfg.variable:
        points = torch.randn(B*N, 3).cuda()
        features = torch.randn(B*N, C).cuda()
        offset = []
        count = 0
        for i in range(B):
            count += N 
            offset.append(count)
        offsets = torch.IntTensor(offset).cuda()
        args = [points, features, offsets]   
    else:
        points = torch.randn(B, N, 3).cuda()
        if cfg.model.get('feature_last_dim', False):
            features = torch.randn(B, N, C).cuda()
        else:
            features = torch.randn(B, C, N).cuda()
        args = [points, features]   
    print(f'test input size: ({points.shape, features.shape})') 
    from deepspeed.profiling.flops_profiler import get_model_profile
    detailed = False
    flops, macs, params = get_model_profile(
        model=model,
        args=args,
        print_profile=detailed,  # prints the model graph with the measured profile attached to each module
        detailed=detailed,  # print the detailed profile
        warm_up=10,  # the number of warm-ups before measuring the time of each module
        as_string=False,  # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
        output_file=None,  # path to the output file. If None, the profiler prints to stdout.
        ignore_modules=None)  # the list of modules to ignore in the profiling
    print(f'GFLOPs\tGMACs\tParams.(M)')
    print(f'{flops/(float(B)*1e9): .2f}\t{macs/(float(B)*1e9): .2f}\t{params/1e6: .3f}')

    if cfg.get('timing', False):
        B = cfg.batch_size
        if cfg.variable:
            points = torch.randn(B*N, 3).cuda()
            features = torch.randn(B*N, C).cuda()
            offset = []
            count = 0
            for i in range(B):
                count += N 
                offset.append(count)
            offsets = torch.IntTensor(offset).cuda()
            args = [points, features, offsets]   
        else:
            points = torch.randn(B, N, 3).cuda()
            features = torch.randn(B, C, N).cuda()
            args = [points, features]   
        model = build_model_from_cfg(cfg.model).cuda()
        n_runs = cfg.get('nruns', 200)
        with torch.no_grad():
            for _ in range(10):  # warm up.
                model(*args)
            start_time = time.time()
            for _ in range(n_runs):
                model(*args)
                torch.cuda.synchronize()
            time_taken = time.time() - start_time
        n_batches = n_runs * B
        print(f'Throughput (ins./s): {float(n_batches) / float(time_taken)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('S3DIS scene segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--detailed', action='store_true')
    args, opts = parser.parse_known_args()

    file_list = []
    for root, dirs, files in os.walk(args.cfg):
        for file in files:
            if file.endswith('yaml') and not file.startswith('default'):
                if 'para' in file or 'opt' in file or 'aug' in file or '_' in file:
                    warnings.warn(f'skip file {file}')
                else:
                    file = os.path.join(root, file)
                    file_list.append(file)

    if len(file_list) == 0 and args.cfg.endswith('yaml'):
        file_list.append(args.cfg)

    for file in file_list:
        cstr = '-'*12
        print(f'{cstr}\n===>loading from {file}')
        cfg = EasyConfig()
        cfg.load(file, recursive=True)
        cfg.update(opts)

        cfg.variable = 'variable' in file
        if cfg.model.get('encoder_args', None) is not None:
            cfg.model.in_channels = cfg.model.encoder_args.in_channels 

        model = build_model_from_cfg(cfg.model).cuda()
        model.eval()
        model_size = cal_model_parm_nums(model)
        print('Number of params: %.4f M' % (model_size / 1e6))
        profile_model(model, cfg)