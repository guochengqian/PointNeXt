#!/usr/bin/env python3
"""Download a PointNeXt checkpoint from Hugging Face Hub and verify SHA-256."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointnext_torch.checkpoints import DEFAULT_REPO_ID, KNOWN_CHECKPOINTS, download_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="modelnet40-pointnext-s-c64",
        help="Known key or Hub filename. Use --list for keys.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--sha256", default=None, help="Expected SHA-256 override")
    parser.add_argument("--token", default=None, help="HF token for private/gated repos")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for key, spec in KNOWN_CHECKPOINTS.items():
            print(f"{key}: {spec.filename} ({spec.expected_metric})")
        return 0

    path = download_checkpoint(
        args.checkpoint,
        repo_id=args.repo_id,
        revision=args.revision,
        output_dir=args.output_dir,
        sha256=args.sha256,
        token=args.token,
    )
    print(f"Downloaded checkpoint: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
