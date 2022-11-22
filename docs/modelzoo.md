# Model Zoo (Pretrained Models)

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
| **Pix4Point** |87.9 / 86.7 | [87.9 / 86.7 ](https://drive.google.com/drive/folders/1VyAWEYZF-nXIp0zIuCqnpFYwVmJjHihR?usp=share_link) | 22.6M | 28.0G | - |



### S3IDS (6-fold) Segmentation

Throughput (TP) is measured with 16 x 15000 points.

|     name     | mIoU/OA/mAcc (Original) | mIoU/OA/mAcc (our released) | #params | FLOPs |  TP  |
|:---:|:---:|:---:|:---:| :---:|:---:|
| PointNet++ | 54.5 / 81.0 / 67.1 | [68.1 / 87.6 / 78.4](https://drive.google.com/drive/folders/1UOM2H9ax_zCM03wl_ZEc3bWtnZJcrggu?usp=sharing) | 1.0M | 7.2G | 186 |
| **PointNeXt-S** |   68.0 / 87.4 / 77.3  |     [68.0 / 87.4 / 77.3](https://drive.google.com/drive/folders/1KTe82Y-I91HDO6ombWDjHRkkdO_F32wx?usp=sharing)      | 0.8M   | 3.6G  | 227 |
| **PointNeXt-B** |   71.5 / 88.8 / 80.2   |     [71.5 / 88.8 / 80.2](https://drive.google.com/drive/folders/1UJj-tvexA74DYSFpNYdPRmMznJ1-tjTO?usp=sharing)      | 3.8M   | 8.8G | 158 |
| **PointNeXt-L** |   73.9 / 89.8 / 82.2   |     [73.9 / 89.8 / 82.2](https://drive.google.com/drive/folders/1VhL1klLDgRVx1O1PN4XN64S3BYkRvmkV?usp=sharing)      | 7.1M   | 15.2G | 115 |
| **PointNeXt-XL** |   74.9 / 90.3 / 83.0   |     [74.9 / 90.3 / 83.0](https://drive.google.com/drive/folders/19n7jmB7NNKiIL_jb3Wq3hY-CQDx23q7u?usp=sharing)      | 41.6M | 84.8G | 46 |



### S3DIS (Area 5) Segmentation

Throughput (TP) is measured with 16 x 15000 points.

|       name       |    mIoU/OA/mAcc (Original)     |                 mIoU/OA/mAcc (our released)                  | #params | FLOPs |  TP  |
|:---:|:---:|:---:|:---:| :---:|:---:|
|    PointNet++    |        53.5 / 83.0 / -         |                    [63.6 / 88.3 / 70.2](https://drive.google.com/drive/folders/1NCy1Av1-TSs_46ngOk181A3BUhc8hpWV?usp=sharing)                    | 1.0M | 7.2G | 186 |
| ASSANet | 63.0 / - /- | [65.8 / 88.9 / 72.2](https://drive.google.com/drive/folders/1a-2yNP_JvOgKPTLBTYXP5NtUmspc1P-c?usp=sharing) | 2.4M | 2.5G | 228 |
| ASSANet-L | 66.8 / - / - | [68.0 / 89.7/ 74.3](https://drive.google.com/drive/folders/1FinOKtFEigsbgjsLybhpZr2xkESLIDhf?usp=sharing) | 115.6M | 36.2G | 81 |
| **PointNeXt-S**  | 63.4±0.8 / 87.9±0.3 / 70.0±0.7 |                    [64.2 / 88.2 / 70.7](https://drive.google.com/drive/folders/1UG8hh_CrUf-OhrYbcDd0zvDtoInrGP1u?usp=sharing)                    |  0.8M   | 3.6G  | 227  |
| **PointNeXt-B**  | 67.3±0.2 / 89.4±0.1 / 73.7±0.6 |                    [67.5 / 89.4 / 73.9](https://drive.google.com/drive/folders/166g_4vaCrS6CSmp3FwAWxl8N8ZmMuylw?usp=sharing)                    |  3.8M   | 8.8G | 158  |
| **PointNeXt-L**  | 69.0±0.5 / 90.0±0.1 / 75.3±0.8 |                    [69.3 / 90.1 / 75.7](https://drive.google.com/drive/folders/1g4qE6g10zoZY5y6LPDQ5g12DvSLbnCnj?usp=sharing)                    |  7.1M   | 15.2G | 115  |
| **PointNeXt-XL** | 70.5±0.3 / 90.6±0.2 / 76.8±0.7 | [71.1 / 91.0 / 77.2](https://drive.google.com/drive/folders/1rng7YmfzzIGtXREn7jW0vVFmSSakLQs4?usp=sharing) |   41.6M | 84.8G | 46  |
| **Pix4Point** | 69.6 / 89.9 / 75.2 | [69.6 / 89.9 / 75.2](https://drive.google.com/drive/folders/1WaJwwWRmv_XtApYKuslPw-hkK0EHvnaE?usp=share_link) |   23.7M | 190G | - |


### ShapeNetpart Part Segmentation

Throughput (TP) is measured with 64*2048 points.

|       name       |    Ins. mIoU / Cat. mIoU (Original)     |                 Ins. mIoU / Cat. mIoU (our released)                  | 
|:---:|:---:|:---:|
|    PointNet++    |        85.1/81.9        |                                        | 1.0M | - | 560 |
| **PointNeXt-S**  | 86.7±0.0 / 84.4±0.2  |                    [86.7 / 84.2](https://drive.google.com/drive/u/1/folders/1SMmQ7EOMXJuBIfjk2T93Pan3T2WaG9Pj?usp=sharing)                    |
| **PointNeXt-S (C=64)**  | 86.9±0.0 / 84.8±0.5 |                    [86.9 / 85.2](https://drive.google.com/drive/u/1/folders/1qGue_R313Ej9xa_VaQGrFHiHoVdD8e8A?usp=sharing)                    |
| **PointNeXt-S (C=160)**  | 87.0±0.1 / 85.2±0.1  |                    [87.1 / 85.4](https://drive.google.com/drive/u/1/folders/1hYxwuAMXo2HRtqEYe0_frjcVD8JDecb6?usp=sharing)                    | 
| **Pix4Point**  | 86.8 / 85.6  |  [86.8 / 85.6](https://drive.google.com/drive/folders/1cHW8phXBB-eOohZ04iUpbo1ulAE2qQkQ?usp=share_link)                    | 



### ModelNet40 Classificaiton

| name | OA/mAcc (Original) |OA/mAcc (our released) | #params | FLOPs | Throughput (ins./sec.) |
|:---:|:---:|:---:|:---:| :---:|:---:|
| PointNet++ | 91.9 / - | [93.0 / 90.7](https://drive.google.com/drive/folders/1Re2_NCtZBKxIhtv755LlnHjz-FBPWjgW?usp=sharing) | 1.5M | 1.7G | 1872 |
| **PointNeXt-S** (C=64) | 93.7±0.3 / 90.9±0.5 | [94.0 / 91.1](https://drive.google.com/drive/folders/14biOHuvH8b2F03ZozrWyF45tCmtsorYN?usp=sharing) | 4.5M | 6.5G | 2033 |
