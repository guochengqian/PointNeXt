# PointNeXt
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pointnext-revisiting-pointnet-with-improved/semantic-segmentation-on-s3dis)](https://paperswithcode.com/sota/semantic-segmentation-on-s3dis?p=pointnext-revisiting-pointnet-with-improved)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pointnext-revisiting-pointnet-with-improved/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=pointnext-revisiting-pointnet-with-improved)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pointnext-revisiting-pointnet-with-improved/semantic-segmentation-on-s3dis-area5)](https://paperswithcode.com/sota/semantic-segmentation-on-s3dis-area5?p=pointnext-revisiting-pointnet-with-improved)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pointnext-revisiting-pointnet-with-improved/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=pointnext-revisiting-pointnet-with-improved)

[arXiv](https://arxiv.org/abs/2206.04670) | [OpenPoints Library](https://github.com/guochengqian/openpoints)

<p align="center">
<img src="misc/effects_training_scaling.png" width=85% height=85% class="center">
</p>

Official PyTorch implementation of PointNeXt, for the following paper:

**PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies**

*by [Guocheng Qian](https://www.gcqian.com/), [Yuchen Li](https://cemse.kaust.edu.sa/vision-cair/people/person/yuchen-li), [Houwen Peng](https://houwenpeng.com/), [Jinjie Mai](https://cemse.kaust.edu.sa/people/person/jinjie-mai), [Hasan Hammoud](https://cemse.kaust.edu.sa/ece/people/person/hasan-abed-al-kader-hammoud), [Mohamed Elhoseiny](http://www.mohamed-elhoseiny.com/), [Bernard Ghanem](https://www.bernardghanem.com/)*

**TL;DR:** We propose improved training and model scaling strategies to boost PointNet++ to state-of-the-art level. PointNet++ with the proposed model scaling is named as PointNeXt, the next version of PointNets. 


<p align="center">
<img src="misc/pointnext.jpeg" width=85% height=85% class="center">
</p>

## News
-  :pushpin:  [Houwen Peng](https://houwenpeng.com/) is hiring research interns at Microsoft Research Asia. Contact: houwen.peng@microsoft.com 
-  :pushpin:  [Bernard Ghanem](https://www.bernardghanem.com/) is hiring visiting students. Monthly salary is paid with free housing. Contact Guocheng if interested: guocheng.qian@kaust.edu.sa
-  2022/06/12: Code released

## Features
In the PointNeXt project, we propose a new and flexible codebase for point-based methods, namely [**OpenPoints**](https://github.com/guochengqian/openpoints). The biggest difference between OpenPoints and other libraries is that we focus more on reproducibility and fair benchmarking. 

1. **Extensibility**: supports many representative networks for point cloud understanding, such as *PointNet, DGCNN, DeepGCN, PointNet++, ASSANet, PointMLP*, and our ***PointNeXt***. More networks can be built easily based on our framework since OpenPoints support a wide range of basic operations including graph convolutions, self-attention, farthest point sampling, ball query, *e.t.c*.

2. **Reproducibility**: all implemented models are trained on various tasks at least three times. Mean±std is provided in the [PointNeXt paper](https://arxiv.org/abs/2206.04670).  *Pretrained models and logs* are available.

3. **Fair Benchmarking**: in PointNeXt, we find a large part of performance gain is due to the training strategies. In OpenPoints, all models are trained with the improved training strategies and all achieve much higher accuracy than the original reported value. 

4. **Ease of Use**: *Build* model, optimizer, scheduler, loss function,  and data loader *easily from cfg*. Train and validate different models on various tasks by simply changing the `cfg\*\*.yaml` file. 

   ```
   model = build_model_from_cfg(cfg.model)
   criterion = build_criterion_from_cfg(cfg.criterion)
   ```
   Here is an example of `pointnet.yaml` (model configuration for PointNet model):
   ```python
   model:
     NAME: BaseCls
     encoder_args:
       NAME: PointNetEncoder
       in_channels: 4
     cls_args:
       NAME: ClsHead
       num_classes: 15
       in_channels: 1024
       mlps: [512,256]
       norm_args: 
         norm: 'bn1d'
   ```

5. **Online logging**: *Support [wandb](https://wandb.ai/)* for checking your results anytime anywhere. 

   ![misc/wandb.png](misc/wandb.png)


## Installation

```
git clone git@github.com:guochengqian/PointNeXt.git
cd PointNeXt
source install.sh
```
Note:  

1) the `install.sh` requires CUDA 11.1; if another version of CUDA is used,  `install.sh` has to be modified accordingly; check your CUDA version by: `nvcc --version` before using the bash file;
2) you might need to read the `install.rst` for a step-by-step installation if the bash file (`install.sh`) does not work for you by any chance;
3) for all experiments, we use wandb for online logging by default. Run `wandb --login` only at the first time in a new machine, or set `wandn.use_wandb=False` if you do not want to use wandb. Read the [official wandb documentation](https://docs.wandb.ai/quickstart) if needed.



## Usage 

**Check `README.md` file under `cfgs` directory for detailed training and evaluation on each benchmark.**  

For example, 
* Train and validate on ScanObjectNN for 3D object classification, check [`cfgs/scanobjectnn/README.md`](cfgs/scanobjectnn/README.md)
* Train and validate on S3DIS for 3D segmentation, check [`cfgs/s3dis/README.md`](cfgs/s3dis/README.md)

Note:  
1. We use *yaml* to support training and validation using different models on different datasets. Just use `.yaml` file accordingly. For example, train on ScanObjectNN using PointNeXt: `CUDA_VISIBLE_DEVICES=1 bash script/main_classification.sh cfgs/scanobjectnn/pointnext-s.yaml`, train on S3DIS using ASSANet-L: `CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/assanet-l.yaml`.  
2. Check the default arguments of each .yaml file. You can overwrite them simply through the command line. E.g. overwrite the batch size, just appending `batch_size=32` or `--batch_size 32`.  


## Model Zoo

We provide the **training logs & pretrained models** in column `our released`  *trained with the improved training strategies proposed by our PointNeXt* through Google Drive. 

*TP*: Throughput (instance per second) measured using an NVIDIA Tesla V100 32GB GPU and a 32 core Intel Xeon @ 2.80GHz CPU.

### ScanObjectNN (Hardest variant) Classification

Throughput is measured with 128 x 1024 points. 

| name | OA/mAcc (Original) |OA/mAcc (our released) | #params | FLOPs | Throughput (ins./sec.) |
|:---:|:---:|:---:|:---:| :---:|:---:|
|  PointNet   | 68.2 / 63.4 | [75.2 / 71.4](https://drive.google.com/drive/folders/1F9sReTX9MC1RAEHZaSh6o_tn9hgQFT95?usp=sharing) | 3.5M | 1.0G | 4212 |
| DGCNN | 78.1 / 73.6 | [86.1 / 84.3](https://drive.google.com/drive/folders/1KWfvYPrJNdOaMOxTQnwI0eXTspKHCfYQ?usp=sharing) | 1.8M | 4.8G | 402 |
| PointMLP |85.4±1.3 / 83.9±1.5 | [87.7 / 86.4](https://drive.google.com/drive/folders/1Cy4tC5YmlbiDATWEW3qLLNxnBx2-XMqa?usp=sharing) | 13.2M | 31.4G | 191 |
| PointNet++ | 77.9 / 75.4 | [86.2 / 84.4](https://drive.google.com/drive/folders/1T7uvQW4cLp65DnaEWH9eREH4XKTKnmks?usp=sharing) | 1.5M | 1.7G | 1872 |
| **PointNeXt-S** |87.7±0.4 / 85.8±0.6 | [88.20 / 86.84](https://drive.google.com/drive/folders/1A584C9x5uAqppbjNNiVqlA_7uOOOlEII?usp=sharing) | 1.4M | 1.64G | 2040 |



### S3IDS (6-fold) Segmentation

Throughput (TP) is measured with 16 x 15000 points.

|     name     | mIoU/OA/mAcc (Original) | mIoU/OA/mAcc (our released) | #params | FLOPs |  TP  |
| :----------: | :------------------: | :-------------------------: | :-----: | :---: | :--: |
| PointNet++ | 54.5 / 81.0 / 67.1 | [68.1 / 87.6 / 78.4](https://drive.google.com/drive/folders/1UOM2H9ax_zCM03wl_ZEc3bWtnZJcrggu?usp=sharing) | 1.0M | 7.2G | 186 |
| **PointNeXt-S** |   68.0 / 87.4 / 77.3  |     [68.0 / 87.4 / 77.3](https://drive.google.com/drive/folders/1KTe82Y-I91HDO6ombWDjHRkkdO_F32wx?usp=sharing)      | 0.8M   | 3.6G  | 227 |
| **PointNeXt-B** |   71.5 / 88.8 / 80.2   |     [71.5 / 88.8 / 80.2](https://drive.google.com/drive/folders/1UJj-tvexA74DYSFpNYdPRmMznJ1-tjTO?usp=sharing)      | 3.8M   | 8.8G | 158 |
| **PointNeXt-L** |   73.9 / 89.8 / 82.2   |     [73.9 / 89.8 / 82.2](https://drive.google.com/drive/folders/1VhL1klLDgRVx1O1PN4XN64S3BYkRvmkV?usp=sharing)      | 7.1M   | 15.2G | 115 |
| **PointNeXt-XL** |   74.9 / 90.3 / 83.0   |     [74.9 / 90.3 / 83.0](https://drive.google.com/drive/folders/19n7jmB7NNKiIL_jb3Wq3hY-CQDx23q7u?usp=sharing)      | 41.6M | 84.8G | 46 |



### S3DIS (Area 5) Segmentation

Throughput (TP) is measured with 16 x 15000 points.

|       name       |    mIoU/OA/mAcc (Original)     |                 mIoU/OA/mAcc (our released)                  | #params | FLOPs |  TP  |
| :--------------: | :----------------------------: | :----------------------------------------------------------: | :-----: | :---: | :--: |
|    PointNet++    |        53.5 / 83.0 / -         |                    [63.6 / 88.3 / 70.2](https://drive.google.com/drive/folders/1NCy1Av1-TSs_46ngOk181A3BUhc8hpWV?usp=sharing)                    | 1.0M | 7.2G | 186 |
| ASSANet | 63.0 / - /- | [65.8 / 88.9 / 72.2](https://drive.google.com/drive/folders/1a-2yNP_JvOgKPTLBTYXP5NtUmspc1P-c?usp=sharing) | 2.4M | 2.5G | 228 |
| ASSANet-L | 66.8 / - / - | [68.0 / 89.7/ 74.3](https://drive.google.com/drive/folders/1FinOKtFEigsbgjsLybhpZr2xkESLIDhf?usp=sharing) | 115.6M | 36.2G | 81 |
| **PointNeXt-S**  | 63.4±0.8 / 87.9±0.3 / 70.0±0.7 |                    [64.2 / 88.2 / 70.7](https://drive.google.com/drive/folders/1UG8hh_CrUf-OhrYbcDd0zvDtoInrGP1u?usp=sharing)                    |  0.8M   | 3.6G  | 227  |
| **PointNeXt-B**  | 67.3±0.2 / 89.4±0.1 / 73.7±0.6 |                    [67.5 / 89.4 / 73.9](https://drive.google.com/drive/folders/166g_4vaCrS6CSmp3FwAWxl8N8ZmMuylw?usp=sharing)                    |  3.8M   | 8.8G | 158  |
| **PointNeXt-L**  | 69.0±0.5 / 90.0±0.1 / 75.3±0.8 |                    [69.3 / 90.1 / 75.7](https://drive.google.com/drive/folders/1g4qE6g10zoZY5y6LPDQ5g12DvSLbnCnj?usp=sharing)                    |  7.1M   | 15.2G | 115  |
| **PointNeXt-XL** | 70.5±0.3 / 90.6±0.2 / 76.8±0.7 | [71.1 / 91.0 / 77.2](https://drive.google.com/drive/folders/1rng7YmfzzIGtXREn7jW0vVFmSSakLQs4?usp=sharing) |   41.6M | 84.8G | 46  |



### ShapeNetpart Part Segmentation

The code and models of ShapeNetPart will come soon.



### ModelNet40 Classificaiton



| name | OA/mAcc (Original) |OA/mAcc (our released) | #params | FLOPs | Throughput (ins./sec.) |
|:---:|:---:|:---:|:---:| :---:|:---:|
| PointNet++ | 91.9 / - | [93.0 / 90.7](https://drive.google.com/drive/folders/1Re2_NCtZBKxIhtv755LlnHjz-FBPWjgW?usp=sharing) | 1.5M | 1.7G | 1872 |
| **PointNeXt-S** (C=64) | 93.7±0.3 / 90.9±0.5 | [94.0 / 91.1](https://drive.google.com/drive/folders/14biOHuvH8b2F03ZozrWyF45tCmtsorYN?usp=sharing) | 4.5M | 6.5G | 2033 |




### Visualization

More examples are available in the [paper](https://arxiv.org/abs/2206.04670). 

![s3dis](misc/s3dis_vis.png)
![shapenetpart](misc/shapenetpart_vis.png)



### Acknowledgment
This library is inspired by [PyTorch-image-models](https://github.com/rwightman/pytorch-image-models) and [mmcv](https://github.com/open-mmlab/mmcv). 



### Citation
If you find PointNeXt or the OpenPoints codebase is useful, please cite:
```tex
@Article{qian2022pointnext,
  author  = {Qian, Guocheng and Li, Yuchen and Peng, Houwen and Mai, Jinjie and Hammoud, Hasan and Elhoseiny, Mohamed and Ghanem, Bernard},
  journal = {arXiv:2206.04670},
  title   = {PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies},
  year    = {2022},
}
```