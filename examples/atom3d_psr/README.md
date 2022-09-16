

## Installation

- Download `PSR` dataset

    ```bash
    python examples/atom3d_psr/download_data.py
    ``` 


## Train 
* from scratch
```bash
python -m torch.distributed.run --nnodes 1 --nproc_per_node 1 examples/atom3d_psr/main_psr_dist.py --cfg cfgs/atom3d_psr/vit.yaml
```

