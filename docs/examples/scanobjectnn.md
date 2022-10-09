# 3D object classification on ScanObjectNN

## Dataset
There are three ways to download the data: 

1. Download from the [official website](https://hkust-vgd.github.io/scanobjectnn/).

2. Or, one can download the data by this command (please submit the  "ScanObjectNN Terms of Use" form on their official website before downloading):
    ```bash
    mkdir -p data/ScanObjectNN
    cd data/ScanObjectNN
    wget http://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip
    ```
    
3. Or, one can only download the hardest variant by the following link. Please cite their paper[1] if you use the link to download the data

    ```bash
    mkdir data
    cd data
    gdown https://drive.google.com/uc?id=1iM3mhMJ_N0x5pytcP831l3ZFwbLmbwzi
    tar -xvf ScanObjectNN.tar
    ```

Organize the dataset as follows:

```
data
 |--- ScanObjectNN
            |--- h5_files
                    |--- main_split
                            |--- training_objectdataset_augmentedrot_scale75.h5
                            |--- test_objectdataset_augmentedrot_scale75.h5
```



## Train

For example, train `PointNext-S`
```bash
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/pointnext-s.yaml
```

* change the cfg file to use any other model, *e.g.* `cfgs/scanobjectnn/pointnet++.yaml` for training PointNet++  



## Test

```bash
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/pointnext-s.yaml  mode=test --pretrained_path pretrained/scanobjectnn/pointnext-s/pointnext-s_best.pth 
```
* change the cfg file to use any other model, *e.g.* `cfgs/scanobjectnn/pointnet++.yaml` for testing PointNet++  



## Profile parameters, FLOPs, and Throughput

```
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/scanobjectnn/pointnext-s.yaml batch_size=128 num_points=1024 timing=True flops=True
```

note: 
1. set `--cfg` to `cfgs/scanobjectnn` to profile all models under the folder. 
2. you have to install the latest version of [DeepSpeed](https://github.com/microsoft/DeepSpeed) from source to get a correct measurement of FLOPs

## Reference

```
@inproceedings{uy-scanobjectnn-iccv19,
      title = {Revisiting Point Cloud Classification: A New Benchmark Dataset and Classification Model on Real-World Data},
      author = {Mikaela Angelina Uy and Quang-Hieu Pham and Binh-Son Hua and Duc Thanh Nguyen and Sai-Kit Yeung},
      booktitle = ICCV,
      year = {2019}
  }
```
