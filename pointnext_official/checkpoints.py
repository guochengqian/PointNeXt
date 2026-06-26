"""Utilities for downloading and verifying PointNeXt checkpoints."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

DEFAULT_REPO_ID = "guochengqian/pointnext"
DEFAULT_MANIFEST = "metadata/checksums.sha256"


@dataclass(frozen=True)
class CheckpointSpec:
    """Description of a released checkpoint and its matching config."""

    key: str
    filename: str
    config: str
    dataset: str
    architecture: str
    expected_metric: str
    sha256: Optional[str] = None
    notes: str = ""


# SHA-256 values are intentionally left unset until the artifacts are staged on
# Hugging Face and metadata/checksums.sha256 is generated from the real files.
KNOWN_CHECKPOINTS: Dict[str, CheckpointSpec] = {
    "modelnet40-pointnext-s-c64": CheckpointSpec(
        key="modelnet40-pointnext-s-c64",
        filename="checkpoints/modelnet40/pointnext-s-c64.pth",
        config="configs/modelnet40/pointnext-s-c64.yaml",
        dataset="ModelNet40Ply2048",
        architecture="PointNeXt-S, width=64, in_channels=3, num_classes=40",
        expected_metric="OA 94.0 / mAcc 91.1 (released checkpoint)",
        notes="Use cfgs/modelnet40ply2048/pointnext-s.yaml with model.encoder_args.width=64.",
    ),
    "scanobjectnn-pointnext-s": CheckpointSpec(
        key="scanobjectnn-pointnext-s",
        filename="checkpoints/scanobjectnn/pointnext-s.pth",
        config="configs/scanobjectnn/pointnext-s.yaml",
        dataset="ScanObjectNN hardest split",
        architecture="PointNeXt-S, width=32, in_channels=4, num_classes=15",
        expected_metric="OA 88.20 / mAcc 86.84 (released checkpoint)",
    ),
}


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hex digest for *path*."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: os.PathLike[str] | str, expected_sha256: str) -> str:
    """Verify *path* against *expected_sha256* and return the actual digest."""

    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    return actual


def parse_sha256_manifest(lines: Iterable[str]) -> Dict[str, str]:
    """Parse a sha256sum-style manifest into {relative_path: digest}."""

    manifest: Dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        digest, filename = parts
        filename = filename.lstrip("*").strip()
        manifest[filename] = digest
    return manifest


def _manifest_digest(repo_id: str, filename: str, revision: Optional[str]) -> Optional[str]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    try:
        manifest_path = hf_hub_download(
            repo_id=repo_id,
            filename=DEFAULT_MANIFEST,
            revision=revision,
            repo_type="model",
        )
    except Exception:
        return None

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = parse_sha256_manifest(handle)
    return manifest.get(filename)


def download_checkpoint(
    key_or_filename: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: Optional[str] = None,
    output_dir: os.PathLike[str] | str = "checkpoints",
    sha256: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """Download a PointNeXt checkpoint from Hugging Face Hub.

    Args:
        key_or_filename: A key from ``KNOWN_CHECKPOINTS`` or a Hub filename such
            as ``checkpoints/modelnet40/pointnext-s-c64.pth``.
        repo_id: Hugging Face model repo ID.
        revision: Optional branch/tag/commit on the Hub repo.
        output_dir: Local directory where the file should be copied/downloaded.
        sha256: Optional expected SHA-256. If omitted, the function tries to read
            ``metadata/checksums.sha256`` from the Hub repo.
        token: Optional Hugging Face token for private/gated repos.

    Returns:
        Path to the downloaded checkpoint.
    """

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Install checkpoint support with `pip install huggingface_hub` or "
            "`pip install pointnext_official`."
        ) from exc

    spec = KNOWN_CHECKPOINTS.get(key_or_filename)
    filename = spec.filename if spec is not None else key_or_filename
    expected = sha256 or (spec.sha256 if spec is not None else None)
    if expected is None:
        expected = _manifest_digest(repo_id, filename, revision)

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        repo_type="model",
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        token=token,
    )
    path = Path(local_path)
    if expected:
        verify_sha256(path, expected)
    return path
