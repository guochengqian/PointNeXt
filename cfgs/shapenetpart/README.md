# Part Segmentation on ShapeNetPart
Note in this experiment, we work on the ShapeNetPart Segmentation. The number of parts for each category is between 2 and 6, with 50 different parts in total. 
The data we use is: `shapenetcore_partanno_segmentation_benchmark_v0_normal.zip`[1]. We uniformly sample 2048 points in training and testing. 


## Dataset
Download the dataset from the official website, put the data under `data/ShapeNetPart/`, and then run the training code once. The data will be autmmatically preprocessed (uniformly sample 2048 points). 

You can also use our preprocessed data provided below:
```
cd data && mkdir ShapeNetPart && cd ShapeNetPart
gdown https://drive.google.com/uc?id=1W3SEE-dY1sxvlECcOwWSDYemwHEUbJIS
tar -xvf shapenetcore_partanno_segmentation_benchmark_v0_normal.tar
```

Organize the dataset as follows:

```
data
 |--- ShapeNetPart
        |--- shapenetcore_partanno_segmentation_benchmark_v0_normal
                |--- train_test_split
                      |--- shuffled_train_file_list.json
                      |--- ...
                |--- 02691156
                      |--- 1a04e3eab45ca15dd86060f189eb133.txt
                      |--- ...               
                |--- 02773838
                |--- synsetoffset2category.txt
                |--- processed
                        |--- trainval_2048_fps.pkl
                        |--- test_2048_fps.pkl
```

## Train
For example, train `PointNeXt-S`
```bash
CUDA_VISIBLE_DEVICES=0 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointnext-s.yaml
```
- change `cfg` to `cfgs/shapenetpart/pointnext-s_c160.yaml` to train the best model we report in our paper.  


## Test
```bash
CUDA_VISIBLE_DEVICES=0 python examples/shapenetpart/main.py cfgs/shapenetpart/pointnext-s.yaml mode=test --pretrained_path /path/to/your/pretrained_model
```


## Profile parameters, FLOPs, and Throughput
CUDA_VISIBLE_DEVICES=1 python examples/profile.py --cfg cfgs/shapenetpart/pointnext-s.yaml batch_size=64 num_points=2048 timing=True flops=True

## Reference
```
@article{yi2016shapnetpart,
Author = {Li Yi and Vladimir G. Kim and Duygu Ceylan and I-Chao Shen and Mengyan Yan and Hao Su and Cewu Lu and Qixing Huang and Alla Sheffer and Leonidas Guibas},
Journal = {SIGGRAPH Asia},
Title = {A Scalable Active Framework for Region Annotation in 3D Shape Collections},
Year = {2016}}`
```