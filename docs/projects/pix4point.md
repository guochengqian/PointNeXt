## Improving Standard Transformer Models for 3D Point Cloud Understanding with Image Pretraining
*by [Guocheng Qian](https://www.gcqian.com/), [Xingdi Zhang](https://cemse.kaust.edu.sa/cs/people/person/xingdi-zhang-0), [ Abdullah Hamdi](https://abdullahamdi.com/), [Bernard Ghanem](https://www.bernardghanem.com/)*
<p align="center">
<img src="../misc/pix4point.png" width=100% height=100% class="center">
</p>



---
### [arXiv](https://arxiv.org/abs/2208.12259) | [code](https://github.com/guochengqian/PointNeXt)

### News
-  :boom: Sep, 2022: code released

### Abstract
While Standard Transformer (ST) models have achieved impressive success in natural language processing and computer vision, their performance on 3D point clouds is relatively poor. This is mainly due to the limitation of Transformers: a demanding need for large training data. Unfortunately, in the realm of 3D point clouds, the availability of large datasets is a challenge, which exacerbates the issue of training ST models for 3D tasks. In this work, we propose two contributions to improve ST models on point clouds. First, we contribute a new ST-based point cloud network, by using Progressive Point Patch Embedding as the tokenizer and Feature Propagation with global representation appending as the decoder. Our network is shown to be less hungry for data, and enables ST to achieve performance comparable to the state-of-the-art. Second, we formulate a simple yet effective pipeline dubbed \textit{Pix4Point}, which allows harnessing Transformers pretrained in the image domain to enhance downstream point cloud understanding. This is achieved through a modality-agnostic ST backbone with the help of our proposed tokenizer and decoder specialized in the 3D domain. Pretrained on a large number of widely available images, we observe significant gains of our ST model in the tasks of 3D point cloud classification, part segmentation, and semantic segmentation on ScanObjectNN, ShapeNetPart, and S3DIS benchmarks, respectively. Our code and models are available at [PointNeXt repo](https://github.com/guochengqian/pointnext). 

## Setup environment
- git clone this repository and install requirements:
  ```bash
  git clone git@github.com:guochengqian/Pix4Point.git
  cd Pix4point
  bash install.sh
  ```

- download the ImageNet21k pretrained Transformer, and put it in `pretrained/imagenet/small_21k_224.pth`
  ```bash
  gdown https://drive.google.com/file/d/1Iqc-nWVMmm4c8kYshNFcJsthnUy75Jl1/view?usp=sharing --fuzzy
  ```


## ImageNet Pretraining
Please refer to DeiT's repo for details. 
 

## Point Cloud Tasks Finetuning
### S3DIS

- finetune Image Pretrained Transformer 
  ```bash
  CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py --cfg cfgs/s3dis_sphere_pix4point/pix4point.yaml
  ```

- test
  ```bash
  CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py --cfg cfgs/s3dis_sphere_pix4point/pix4point.yaml mode=test  pretrained_path=<pretrained_path>
  ```


### ScanObjectNN

- finetune 
  ```bash
  CUDA_VISIBLE_DEVICES=0  python examples/classification/main.py --cfg cfgs/scanobjectnn_pix4point/pvit.yaml
  ``` 


### ModelNet40  
<!-- - scratch 
  ```bash
  CUDA_VISIBLE_DEVICES=0python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pix4point.yaml mode=train 
  ``` -->

- finetune 
  ```bash
  CUDA_VISIBLE_DEVICES=0  python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pix4point.yaml
  ``` 

  
### ShapeNetPart
- finetune 
  ```bash
  CUDA_VISIBLE_DEVICES=0  python examples/shapenetpart/main.py --cfg cfgs/shapenetpart_pix4point/pix4point.yaml 
  ```


  
## Citation
If you are using our code in your work, please kindly cite the following:  
```
@misc{qian2022improving,
    title={Improving Standard Transformer Models for 3D Point Cloud Understanding with Image Pretraining},
    author={Guocheng Qian and Xingdi Zhang and Abdullah Hamdi and Bernard Ghanem},
    year={2022},
    eprint={2208.12259},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
``` 


