"""Command-line helpers for the pointnext-torch package."""

from __future__ import annotations

import argparse

from .checkpoints import DEFAULT_REPO_ID, KNOWN_CHECKPOINTS, download_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download PointNeXt checkpoints from Hugging Face Hub")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="modelnet40-pointnext-s-c64",
        help="Known checkpoint key or Hub filename. Use --list to show known keys.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face model repo ID")
    parser.add_argument("--revision", default=None, help="Optional Hub revision")
    parser.add_argument("--output-dir", default="checkpoints", help="Local output/cache directory")
    parser.add_argument("--sha256", default=None, help="Expected SHA-256 override")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token")
    parser.add_argument("--list", action="store_true", help="List known checkpoint keys and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        for key, spec in KNOWN_CHECKPOINTS.items():
            print(
                f"{key}\n"
                f"  file: {spec.filename}\n"
                f"  config: {spec.config}\n"
                f"  expected: {spec.expected_metric}"
            )
        return 0
    path = download_checkpoint(
        args.checkpoint,
        repo_id=args.repo_id,
        revision=args.revision,
        output_dir=args.output_dir,
        sha256=args.sha256,
        token=args.token,
    )
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
