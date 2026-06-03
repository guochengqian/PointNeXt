# FAQ and common installation/runtime questions

## I cloned the repo but `openpoints` is missing

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/guochengqian/PointNeXt.git
cd PointNeXt
git submodule update --init --recursive
```

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

For a pure Python import check, the OpenPoints library can also be installed with:

```bash
pip install openpoints
```

For training/evaluation, build the CUDA/C++ extensions from a source checkout.

## Does `pip install openpoints pointnext-torch` include CUDA ops?

No. The PyPI packages make the Python modules importable and provide metadata/checkpoint helpers. PointNeXt training/evaluation still uses custom CUDA/C++ operators that depend on the local Python, PyTorch, CUDA, compiler, and platform ABI. Build them from source:

```bash
cd openpoints/cpp/pointnet2_batch && python setup.py install && cd ../../..
cd openpoints/cpp/pointops && python setup.py install && cd ../../..
```

`chamfer_dist` and `emd` are optional for classification/segmentation and mainly needed for reconstruction/completion tasks.

## Can I run PointNeXt on CPU only?

CPU-only import and packaging smoke tests are supported. The main PointNeXt models rely on CUDA custom ops such as ball query / grouping / pointops for practical training and evaluation, so full benchmark reproduction should be run on a CUDA GPU.

## What does `in_channels` mean?

`in_channels` is the number of per-point input feature channels consumed by the encoder.

Examples:

- ModelNet40 PointNeXt-S uses xyz only: `in_channels=3`.
- ScanObjectNN PointNeXt-S uses xyz plus an extra feature/height channel in this config: `in_channels=4`.
- Segmentation configs may use xyz plus color/height/features depending on the dataset pipeline.

A checkpoint must be evaluated with a config that matches its `in_channels`, width, number of classes, and dataset preprocessing.

## Why does ModelNet40 testing use `model.encoder_args.width=64`?

The default `cfgs/modelnet40ply2048/pointnext-s.yaml` is PointNeXt-S width 32. The released ModelNet40 model-zoo checkpoint is the C=64 variant, so testing that checkpoint requires:

```bash
model.encoder_args.width=64
```

Training the default width-32 config does not need this override.

## Headless server visualization crashes or opens no window

Visualization utilities may require an OpenGL context. On remote/headless servers, prefer offscreen rendering or run through a virtual display:

```bash
export PYVISTA_OFF_SCREEN=true
xvfb-run -s "-screen 0 1024x768x24" python examples/segmentation/vis_results.py ...
```

If visualization still segfaults, first verify the non-visual evaluation command on the same checkpoint/config, then report the OS, GPU driver, CUDA, PyTorch, PyVista, and OpenGL/Mesa versions.

## `Permission denied` when running a Python file

Run Python scripts through Python, not as shell executables:

```bash
python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml
```

If a shell script fails with permission denied, either run it with `bash script.sh` or mark it executable with `chmod +x script.sh`.
