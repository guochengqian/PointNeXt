## Pix4Point: Image Pretrained Transformers for 3D Point Cloud Understanding
*by [Guocheng Qian](https://www.gcqian.com/), [Xingdi Zhang](https://cemse.kaust.edu.sa/cs/people/person/xingdi-zhang-0), [ Abdullah Hamdi](https://abdullahamdi.com/), [Bernard Ghanem](https://www.bernardghanem.com/)*
<p align="center">
<img src="../misc/pointnext.jpeg" width=100% height=100% class="center">
</p>



---
### [arXiv](https://arxiv.org/abs/2208.12259) | [OpenPoints Library](https://github.com/guochengqian/openpoints)

### News


### Abstract
Pure Transformer models have achieved impressive success in natural language processing and computer vision. However, one limitation with Transformers is their need for large training data. In the realm of 3D point clouds, the availability of large datasets is a challenge, which exacerbates the issue of training Transformers for 3D tasks. In this work, we empirically study and investigate the effect of utilizing knowledge from a large number of images for point cloud understanding. We formulate a pipeline dubbed Pix4Point that allows harnessing pretrained Transformers in the image domain to improve downstream point cloud tasks. This is achieved by a modality-agnostic pure Transformer backbone with the help of tokenizer and decoder layers specialized in the 3D domain. Using image-pretrained Transformers, we observe significant performance gains of Pix4Point on the tasks of 3D point cloud classification, part segmentation, and semantic segmentation on ScanObjectNN, ShapeNetPart, and S3DIS benchmarks, respectively. 

## Setup environment
git clone this repository and install requirements:
```
git clone git@github.com:guochengqian/Pix4Point.git
cd Pix4point
bash install.sh
```
## Data preparation
ImageNet-1k dataset is used to pretrain Pix4point backbone.
You could download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:
```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

## S3DIS
- scratch 
  ```bash
CUDA_VISIBLE_DEVICES=1 python examples/segmentation/main.py --cfg cfgs/s3dis_sphere_pix4point/pix4point.yaml mode=train 
  ```

- finetune 
  ```bash
CUDA_VISIBLE_DEVICES=1  python examples/segmentation/main.py --cfg cfgs/s3dis_sphere_pix4point/pix4point.yaml mode=finetune  pretrained_path="pretrained vit encoder path"
  ```
- test
  ```bash
CUDA_VISIBLE_DEVICES=1  python examples/segmentation/main.py --cfg cfgs/s3dis_sphere_pix4point/pix4point.yaml mode=test  pretrained_path="pretrained model path"
  ```
  
  
## ModelNet40  
- scratch 
  ```bash
  CUDA_VISIBLE_DEVICES=1 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pix4point.yaml mode=train 
  ```

- finetune 
  ```bash
CUDA_VISIBLE_DEVICES=1   python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pix4point.yaml  mode=finetune pretrained_path="pretrained vit encoder path"
  ``` 
- test
  ```bash
CUDA_VISIBLE_DEVICES=1  python examples/classification/main.py --cfg cfgs/s3dis_sphere_pix4point/pix4point.yaml mode=test  pretrained_path="pretrained model path"
  ```
  
## ShapeNetPart
- scratch 
  ```bash
CUDA_VISIBLE_DEVICES=1   python examples/shapenetpart/main.py --cfg cfgs/shapenetpart_pix4point/pix4point.yaml  mode=train 
  ```

- finetune 
  ```bash
CUDA_VISIBLE_DEVICES=1   python examples/shapenetpart/main.py --cfg cfgs/shapenetpart_pix4point/pix4point.yaml  mode=finetune  pretrained_path="pretrained vit encoder path"
  ```
- test
  ```bash
CUDA_VISIBLE_DEVICES=1  python examples/shapenetpart/main.py --cfg cfgs/s3dis_sphere_pix4point/pix4point.yaml mode=test  pretrained_path="pretrained model path"
  ```
## ScanObjectNN
- scratch 
  ```bash
CUDA_VISIBLE_DEVICES=1   python examples/classification/main.py --cfg cfgs/scanobjectnn_pix4point/pvit.yaml mode=train 
  ```

- finetune 
  ```bash
CUDA_VISIBLE_DEVICES=1   python examples/classification/main.py --cfg cfgs/scanobjectnn_pix4point/pvit.yaml  pretrained_path="pretrained vit encoder path"
  ```
  
- test
  ```bash
CUDA_VISIBLE_DEVICES=1  python examples/classification/main.py --cfg cfgs/scanobjectnn_pix4point/pvit.yaml mode=test  pretrained_path="pretrained model path"
  ```
  
### Citation
If you are using our code in your work, please kindly cite the following:  
```
@inproceedings{qian2022pix4point,
  title={Pix4Point: Image Pretrained Transformers for 3D Point Cloud Understanding},
  author={Guocheng Qian, Xingdi Zhang, Abdullah Hamdi, Bernard Ghanem},
  publisher = {arXiv},
  year={2022}
}```  


