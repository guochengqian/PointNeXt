"""PointNeXt release helper package.

The research models and training/evaluation code live in the PointNeXt
repository and the OpenPoints package.  This small package provides stable
paths, metadata, and checkpoint download helpers for pip-installed users.
"""

from .checkpoints import (
    DEFAULT_REPO_ID,
    KNOWN_CHECKPOINTS,
    CheckpointSpec,
    download_checkpoint,
    verify_sha256,
)

__all__ = [
    "DEFAULT_REPO_ID",
    "KNOWN_CHECKPOINTS",
    "CheckpointSpec",
    "download_checkpoint",
    "verify_sha256",
]

__version__ = "0.1.0"
