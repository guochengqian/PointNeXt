## Dataset

```
cd data
gdown https://drive.google.com/uc?id=1uWlRPLXocqVbJxPvA2vcdQINaZzXf1z_
tar -xvf ScanNet.tar
```



## Train

For example, train `PointNext-XL`
```bash
python examples/segmentation/main.py --cfg cfgs/scannet/pointnext-xl0.yaml 
```
* change the cfg file to use any other model, *e.g.* `cfgs/s3dis/pointnet++.yaml` for training PointNet++  
* run the command at the root directory

