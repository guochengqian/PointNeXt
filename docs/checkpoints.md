# Checkpoints and Hugging Face release layout

PointNeXt pretrained checkpoints remain available through the links in the Model Zoo. For new releases, the recommended primary/mirror layout is a Hugging Face model repo:

```text
guochengqian/pointnext
  README.md
  checkpoints/
    modelnet40/pointnext-s-c64.pth
    scanobjectnn/pointnext-s.pth
    s3dis/pointnext-xl-area5.pth
    scannet/pointnext-s-val.pth
  configs/
    modelnet40/pointnext-s-c64.yaml
    scanobjectnn/pointnext-s.yaml
    s3dis/pointnext-xl.yaml
    scannet/pointnext-s.yaml
  metadata/
    checksums.sha256
```

Each checkpoint should be uploaded with the exact config needed to load it. The checkpoint architecture must match the config, especially:

- `model.encoder_args.width`
- `model.encoder_args.in_channels`
- `cls_args.num_classes` or segmentation class count
- dataset split and number of points

## Download with the helper

After `pointnext-torch` is installed, a known checkpoint can be downloaded with:

```bash
pip install pointnext-torch
pointnext-download modelnet40-pointnext-s-c64 --output-dir ./hf_cache
```

From a source checkout, the same helper is available as:

```bash
python tools/download_checkpoint.py modelnet40-pointnext-s-c64 --output-dir ./hf_cache
```

The helper uses `huggingface_hub.hf_hub_download`, then verifies SHA-256 if the Hub repo contains `metadata/checksums.sha256` or if `--sha256` is supplied.

For private/gated staging repos, authenticate first:

```bash
hf auth login
# or
export HF_TOKEN=hf_...
python tools/download_checkpoint.py modelnet40-pointnext-s-c64 --token "$HF_TOKEN"
```

## Generate checksums before upload

Stage the Hugging Face repo locally, then generate the manifest from the real files:

```bash
python tools/write_checkpoint_manifest.py /path/to/pointnext-hf-staging
```

This writes:

```text
/path/to/pointnext-hf-staging/metadata/checksums.sha256
```

Do not invent or hand-write checkpoint hashes. Regenerate the manifest whenever an artifact changes.

## Upload to Hugging Face Hub

```bash
hf auth login
hf repos create guochengqian/pointnext --type model
hf upload-large-folder guochengqian/pointnext /path/to/pointnext-hf-staging
```

If `guochengqian/pointnext` already exists, upload into the existing repo. Keep Google Drive links as a fallback/mirror until all public users have a working HF route.

## ModelNet40 PointNeXt-S C=64 example

The released ModelNet40 result in the model zoo is PointNeXt-S with width 64:

```bash
python tools/download_checkpoint.py modelnet40-pointnext-s-c64 --output-dir ./hf_cache
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py   --cfg cfgs/modelnet40ply2048/pointnext-s.yaml   model.encoder_args.width=64   mode=test   --pretrained_path hf_cache/checkpoints/modelnet40/pointnext-s-c64.pth   wandb.use_wandb=False
```

Expected released checkpoint result: about OA 94.0 / mAcc 91.1.
