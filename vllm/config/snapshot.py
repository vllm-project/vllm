# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .utils import config


@config
class SnapshotConfig:
    """Configuration for container snapshot lifecycle management."""

    snapshot_metadata: str | None = None
    """Snapshot metadata file used for checkpoint and restore."""
