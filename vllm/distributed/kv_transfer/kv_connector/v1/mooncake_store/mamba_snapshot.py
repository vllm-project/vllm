# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MambaSnapshotConfig:
    """Store-side retention of exact Mamba boundary states."""

    interval_blocks: int = 1
    keep_prompt_boundary: bool = False

    @classmethod
    def from_extra_config(cls, extra_config: dict[str, Any]) -> "MambaSnapshotConfig":
        interval = extra_config.get("mamba_snapshot_interval_blocks", 1)
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError(
                "mamba_snapshot_interval_blocks must be a positive integer"
            )
        keep = extra_config.get("mamba_snapshot_keep_prompt_boundary", False)
        if isinstance(keep, str):
            value = keep.strip().lower()
            if value in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
                keep = value in {"1", "true", "yes", "on"}
        if not isinstance(keep, bool):
            raise ValueError("mamba_snapshot_keep_prompt_boundary must be a boolean")
        return cls(interval_blocks=interval, keep_prompt_boundary=keep)

    def interval_tokens(self, scheduler_block_size: int) -> int | None:
        return (
            self.interval_blocks * scheduler_block_size
            if self.interval_blocks > 1
            else None
        )
