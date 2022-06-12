# Indoor Point cloud Segmentation on S3DIS
The models are trained on the **subsampled** point clouds (voxel size = 0.04). The model achieving the best performance on validation is selected to test on the **original** point clouds (not downsampled). 



## Dataset

Please cite the S3DIS paper [1] if you are going to use our presampled datasets.  The presampling is just to collect all point clouds, area by area and room by room, following [PointGroup](https://github.com/dvlab-research/PointGroup).

```bash
mkdir -p data/S3DIS/
cd data/S3DIS
gdown https://drive.google.com/uc?id=1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y
tar -xvf s3dis.tar
```

Organize the dataset as follows:

```
data
 |--- S3DIS
        |--- s3disfull
                |--- raw
                      |--- Area_6_pantry_1.npy
                      |--- ...
                |--- processed
                      |--- s3dis_val_area5_0.040.pkl 
```


## Train

For example, train `PointNext-XL`
```bash
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnext-xl.yaml
```
* change the cfg file to use any other model, *e.g.* `cfgs/scanobjectnn/pointnet++.yaml` for training PointNet++  
* run the command at the root directory


## Test on Area 5
Note **testing is a must step** since evaluation in training is performed only on subsampled point clouds not original point clouds. 

```bash
CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/pointnext-xl.yaml wandb.use_wandb=False mode=test --pretrained_path pretrained/s3dis/pointnext-xl/pointnext-xl-area5/checkpoint/pointnext-xl_ckpt_best.pth
```
* add `visualize=True` to save segmentation results as .obj files


## Test on All Areas

```bash
CUDA_VISIBLE_DEVICES=1 python examples/segmentation/test_6fold.py cfgs/s3dis/pointnext-xl.yaml mode=test --pretrained_path pretrained/s3dis/pointnext-xl
```



## Profile Parameters, FLOPs, and Throughput

```bash
CUDA_VISIBLE_DEVICES=1 python examples/profile.py --cfg cfgs/s3dis/pointnext-xl.yaml batch_size=16 num_points=15000 timing=True
```
note: 
1. set `--cfg` to `cfgs/s3dis` to profile all models under the folder.
2. you have to install the latest version of [DeepSpeed](https://github.com/microsoft/DeepSpeed) from source to get a correct measurement of FLOPs



## Reference 

```
@inproceedings{armeni2016s3dis,
  title={3d semantic parsing of large-scale indoor spaces},
  author={Armeni, Iro and Sener, Ozan and Zamir, Amir R and Jiang, Helen and Brilakis, Ioannis and Fischer, Martin and Savarese, Silvio},
  booktitle=CVPR,
  pages={1534--1543},
  year={2016}
}
```
