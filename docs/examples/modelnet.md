# Point cloud classification on ModelNet40

Note in this experiment, we do not use any re-sampled version of ModelNet40 (more than 2K points) or any normal information.  The data we use is: `modelnet40_ply_hdf5_2048`[1].  

## Dataset
ModelNet40 dataset will be downloaded automatically.


## Train
For example, train `PointNeXt-S`
```bash
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml
```

## Test

The default `cfgs/modelnet40ply2048/pointnext-s.yaml` config uses PointNeXt-S with width 32. The released ModelNet40 model-zoo checkpoint is the C=64 variant, so testing that checkpoint requires `model.encoder_args.width=64`.

Download the checkpoint with the helper after the HF mirror is published:

```bash
python tools/download_checkpoint.py modelnet40-pointnext-s-c64 --output-dir ./hf_cache
```

Test `PointNeXt-S (C=64)`:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py \
  --cfg cfgs/modelnet40ply2048/pointnext-s.yaml \
  model.encoder_args.width=64 \
  mode=test \
  --pretrained_path hf_cache/checkpoints/modelnet40/pointnext-s-c64.pth \
  wandb.use_wandb=False
```

Expected released checkpoint result: about OA 94.0 / mAcc 91.1.

## Reference 
```
@InProceedings{wu2015modelnet,
	author = {Wu, Zhirong and Song, Shuran and Khosla, Aditya and Yu, Fisher and Zhang, Linguang and Tang, Xiaoou and Xiao, Jianxiong},
	title = {3D ShapeNets: A Deep Representation for Volumetric Shapes},
	booktitle = {CVPR},
	year = {2015}
}
```
