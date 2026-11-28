# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .utils import config


@config
class SnapshotConfig:
    """Configuration for container snapshot lifecycle management."""

    snapshot_metadata: str | None = None
    """Snapshot metadata file used for checkpoint and restore."""

    enable_auto_checkpoint: bool = False
    """Whether to manage the checkpoint lifecycle with the snapshot sentinel."""

    def __post_init__(self) -> None:
        if self.enable_auto_checkpoint and self.snapshot_metadata is None:
            raise ValueError(
                "snapshot_metadata is required when enable_auto_checkpoint is true"
            )
