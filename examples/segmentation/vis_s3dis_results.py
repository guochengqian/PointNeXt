import __init__
import os, os.path as osp, numpy as np
from glob import glob
from openpoints.dataset.vis3d import read_obj, vis_multi_points, vis_points

#### names change the name
methods_dir = [
    'pretrained/s3dis/PointNet++original/s3dis-train-pointnet++original-ngpus1-seed5123-20220515-214346-kKe7d9yNdMrnUoLxsYcpCm/visualization', 
    'pretrained/s3dis/PointNeXt-XL/s3disfull-train-pointnext-XL-ngpus1-seed4464-batch_size=8-20220425-014658-7Ew4sJwdKppnWsS6UW2LrA/visualization'
    ]
prefix_names = ['input-Area5-', 'gt-Area5-', 'pred-Area5-']
idx =  45 
# good: 3, 4, 17, 40, 55
# fair: 30
# 65
file_list = glob(osp.join(methods_dir[0], '*'))

save_dir = 'pretrained/s3dis/visualization/'
os.makedirs(save_dir, exist_ok=True) 
save_name = f'{save_dir}/pointnet++_pointnext_area5_{idx}'

input_points, input_colors =read_obj(osp.join(methods_dir[0], prefix_names[0]+str(idx)+'.obj'))
# valid_idx = np.arange(len(input_points))
valid_idx = input_points[:, 2] < 3
input_points = input_points[valid_idx]
gt_points, gt_colors =read_obj(osp.join(methods_dir[0], prefix_names[1]+str(idx)+'.obj'))
method1_points, method1_colors =read_obj(osp.join(methods_dir[0], prefix_names[2]+str(idx)+'.obj'))
method2_points, method2_colors =read_obj(osp.join(methods_dir[1], prefix_names[2]+str(idx)+'.obj'))
vis_multi_points([input_points, input_points, input_points, input_points], [input_colors[valid_idx],  method1_colors[valid_idx], method2_colors[valid_idx], gt_colors[valid_idx]])