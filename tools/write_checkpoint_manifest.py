#!/usr/bin/env python3
"""Generate a sha256sum-style manifest for staged PointNeXt checkpoint files."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

DEFAULT_PATTERNS = ("*.pth", "*.pt", "*.ckpt", "*.safetensors", "*.yaml")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path, patterns: Iterable[str]):
    seen = set()
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Staged HF repo directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output manifest path. Defaults to <root>/metadata/checksums.sha256",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        default=None,
        help="Glob pattern to include; may be repeated. Defaults to checkpoint/config patterns.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    output = args.output or (root / "metadata" / "checksums.sha256")
    patterns = args.patterns or list(DEFAULT_PATTERNS)

    lines = []
    for path in iter_files(root, patterns):
        if output.resolve() == path.resolve():
            continue
        rel = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path)}  {rel}\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {len(lines)} entries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
