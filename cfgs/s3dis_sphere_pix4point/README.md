# Indoor Point cloud Segmentation on S3DIS
The models are trained on **sphere subsampled** point clouds (voxel size = 0.04, radius=2). The model achieving the best performance on validation is selected to test on the **original** point clouds. The testing input is the spere subsampled points, testing results for original point clouds are by nearest neighbor interpolated.


## Dataset

```bash
mkdir -p data/S3DIS/
cd data/S3DIS
gdown https://drive.google.com/uc?id=1dkTba7fqjrkuVslA6EKiUuqeIkbAGZ5A
```

Organize the dataset as follows:

```
data
 |--- S3DIS
        |--- Stanford3dDataset_v1.2 
                  |--- processed
                  |--- Area_1
                  |--- Area_2
                  |--- Area_3
                  |--- Area_4
                  |--- Area_5
                  |--- Area_6
```

## Train

For example, finetune imagenet pretrained `pvit` on S3DIS:
```bash
CUDA_VISIBLE_DEVICES=1 python examples/segmentation/main.py --cfg cfgs/s3dis_sphere_pix4point/pix4point.yaml --pretrained_path xxxx
```


## Reference 
