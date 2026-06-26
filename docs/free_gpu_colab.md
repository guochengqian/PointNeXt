# Free GPU smoke/evaluation

A free GPU can test a small PointNeXt release path, but not every benchmark.

## Good free-GPU targets

- Google Colab Free, usually T4/K80/P100 depending on availability.
- Kaggle Notebook GPU, often T4/P100.
- Similar free notebook environments with one CUDA GPU and a working compiler.

These are suitable for:

- `pip install pointnext_official` import checks.
- source checkout with CUDA op build.
- ModelNet40 dataset auto-download.
- ModelNet40 PointNeXt-S C=64 checkpoint download and SHA-256 verification.
- a single ModelNet40 test/smoke run.

## Not realistic on free GPU

- full S3DIS or ScanNet training
- multi-GPU training
- exhaustive segmentation benchmarking
- large ablation sweeps

## Colab notebook

Use:

```text
notebooks/colab_free_modelnet40_smoke_eval.ipynb
```

The notebook performs:

1. GPU check.
2. clone with submodules.
3. install source dependencies.
4. build PointNeXt/OpenPoints CUDA ops.
5. download the ModelNet40 PointNeXt-S C=64 checkpoint from Hugging Face Hub.
6. verify SHA-256 when `metadata/checksums.sha256` is available.
7. run the ModelNet40 evaluation command.

Expected released checkpoint result: about OA 94.0 / mAcc 91.1. For a quick smoke test, successful import, CUDA op build, checkpoint download/hash verification, dataset load, and a completed evaluation loop are the pass criteria.
