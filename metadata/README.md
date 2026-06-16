# PointNeXt release metadata

`checksums.sha256` is generated from the staged Hugging Face release directory after real checkpoint artifacts are present:

```bash
python tools/write_checkpoint_manifest.py /path/to/pointnext-hf-staging
```

Do not hand-write hashes. The downloader reads this manifest from the Hugging Face Hub repo and verifies downloaded checkpoints before reporting success.
