# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from pydantic import Field

from vllm.config.utils import config


@config
class KVControlConfig:
    """Opt-in engine-local programmable KV control. None disables the feature."""

    engine_id: str | None = Field(default=None, min_length=1, max_length=240)
    """Stable engine name; DP ranks append their rank. None generates a name."""
    attached_consumer: Literal["native", "backend"] = "native"
    """Interpretation owner for the entire request-attached envelope."""
    max_records: int = Field(default=4096, gt=0)
    """Maximum retained operation records, including active operations."""
    max_actions: int = Field(default=128, gt=0)
    """Maximum actions admitted in one envelope."""
    max_envelope_bytes: int = Field(default=1 << 20, gt=0)
    """Maximum serialized control payload size."""
    max_events: int = Field(default=4096, gt=0)
    """Maximum replayable events per operation or association stream."""
    result_ttl: float = Field(default=300.0, gt=0, allow_inf_nan=False)
    """Minimum terminal-result retention in seconds."""
    max_sessions: int = Field(default=4096, gt=0)
    """Maximum tracked sessions."""
    max_continuations: int = Field(default=16384, gt=0)
    """Maximum tracked continuation branches."""
    max_associations: int = Field(default=262144, gt=0)
    """Maximum tracked hash/group/span associations."""
    max_tombstones: int = Field(default=16384, gt=0)
    """Maximum retained explicit cleanup records."""
    max_declarations: int = Field(default=4096, gt=0)
    """Maximum independent retention declarations."""
    max_snapshot_entries: int = Field(default=262144, gt=0)
    """Maximum content entries captured by retention snapshots."""
