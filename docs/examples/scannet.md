# Large-Scale 3D Segmentation on ScanNet


## Dataset


You can download our preprocessed ScanNet dataset as follows:
```bash
cd data
gdown https://drive.google.com/uc?id=1uWlRPLXocqVbJxPvA2vcdQINaZzXf1z_
tar -xvf ScanNet.tar
```
Please cite the ScanNet paper [1] if you are going to conduct experiments on it.



## Train

For example, train `PointNext-XL` using 8 GPUs by default.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/segmentation/main.py --cfg cfgs/scannet/pointnext-xl.yaml 
```
* change the cfg file to use any other model, *e.g.* `cfgs/s3dis/pointnet++.yaml` for training PointNet++  
* run the command at the root directory


## Val 

```bash
CUDA_VISIBLE_DEVICES=0  python examples/segmentation/main.py --cfg cfgs/scannet/<YOUR_CONFIG> mode=test dataset.test.split=val --pretrained_path <YOUR_CHECKPOINT_PATH>
```

## Test

You can generate Scannet benchmark submission file as follows
```bash
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py --cfg cfgs/scannet/<YOUR_CONFIG> mode=test dataset.test.split=test no_label=True pretrained_path=<YOUR_CHECKPOINT_PATH>
```
Please make sure your checkpoint and your cfg matches with each.


## Reference

```
@inproceedings{dai2017scannet,
	title={{ScanNet}: Richly-annotated {3D} Reconstructions of Indoor Scenes},
	author={Dai, Angela and Chang, Angel X. and Savva, Manolis and Halber, Maciej and Funkhouser, Thomas and Nie{\ss}ner, Matthias},
	booktitle = CVPR,
	year = {2017}
}
```
