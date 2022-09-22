

## S3DIS
### Dataset preparation 



- scratch 
  ```bash
  python examples/segmentation/main.py --cfg cfgs/s3dis_sphere_pix4point/pix4point.yaml mode=train pretrained_path=None
  ```

- finetune 
  ```bash
  python examples/segmentation/main.py --cfg cfgs/s3dis_sphere_pix4point/pix4point.yaml  pretrained_path=pretrained/imagenet/small_224.pth
  ```
  - set pretrained_path to load a different pretrained checkpoint to finetune


