# Point cloud classification on ModelNet40

Note in this experiment, we do not use any re-sampled version of ModelNet40 (more than 2K points) or any normal information.  The data we use is: `modelnet40_ply_hdf5_2048`[1].  

## Dataset
ModelNet40 dataset will be downloaded automatically.


## Train
For example, train `PointNeXt-S`
```bash
CUDA_VISIBLE_DEVICES=1 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml
```

## Test

test `PointNeXt-S (C=64)`

```bash
CUDA_VISIBLE_DEVICES=1 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml model.encoder_args.width=64 wandb.use_wandb=False mode=test --pretrained_path /path/to/your/pretrained_model
```

## Reference 
```
@InProceedings{wu2015modelnet,
	author = {Wu, Zhirong and Song, Shuran and Khosla, Aditya and Yu, Fisher and Zhang, Linguang and Tang, Xiaoou and Xiao, Jianxiong},
	title = {3D ShapeNets: A Deep Representation for Volumetric Shapes},
	booktitle = {CVPR},
	year = {2015}
}
```