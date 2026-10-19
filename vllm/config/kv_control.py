# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from pydantic import Field

from vllm.config.utils import config


@config
class KVPolicyConfig:
    name: str = Field(default="retention", min_length=1, max_length=128)
    """Unique policy name used in derived operation results."""
    implementation: str = "retention"
    """Built-in name or importable policy class qualified name."""
    options: dict[str, Any] = Field(default_factory=dict)
    """Constructor options for this policy instance."""


@config
class KVControlConfig:
    """Opt-in engine-local programmable KV control. None disables the feature."""

    engine_id: str | None = Field(default=None, min_length=1, max_length=240)
    """Stable engine name; DP ranks append their rank. None generates a name."""
    attached_consumer: Literal["native", "backend"] = "native"
    """Interpretation owner for the entire request-attached envelope."""
    max_records: int = Field(default=4096, gt=0)
    """Maximum retained operation records, including active operations."""
    max_acknowledged_messages: int = Field(default=65536, gt=0)
    """Maximum compact replay guards after callers acknowledge terminal results."""
    max_actions: int = Field(default=128, gt=0)
    """Maximum actions admitted in one envelope."""
    max_envelope_bytes: int = Field(default=1 << 20, gt=0)
    """Maximum serialized control payload size."""
    max_events: int = Field(default=4096, gt=0)
    """Maximum replayable events per operation or association stream."""
    max_result_targets: int = Field(default=16384, gt=0)
    """Maximum per-target outcomes, checked before physical allocation."""
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
    policies: list[KVPolicyConfig] = Field(default_factory=list)
    """Startup policy instances. Empty keeps native signals unconsumed."""
    composer: str = "priority"
    """Built-in composer name or importable composer class qualified name."""
    policy_interval_seconds: float = Field(default=0.05, gt=0, allow_inf_nan=False)
    """Minimum interval between optional policy observations."""
    max_pending_policy_actions: int = Field(default=256, gt=0)
    """Maximum scheduled proposals, including future prefetch predictions."""
    max_policy_actions_per_step: int = Field(default=64, gt=0)
    """Maximum policy proposals executed on one engine step."""
    max_policy_requests: int = Field(default=4096, gt=0)
    """Maximum requests in one policy observation; overflow disables guidance."""
    max_policy_target_entries: int = Field(default=16384, gt=0)
    """Maximum content entries held by pending policy snapshots."""
    max_policy_wait_seconds: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    """Maximum additional admission/order deferral imposed by local policies."""
    max_policy_delay_seconds: float = Field(default=300.0, gt=0, allow_inf_nan=False)
    """Maximum time a policy may defer execution of a predicted operation."""
    policy_decode_lookahead_tokens: int = Field(default=0, ge=0)
    """Additional remaining output tokens included in quota demand estimates."""
    max_inflight_load_blocks: int = Field(default=0, ge=0)
    """Cap speculative load reservations; zero uses existing physical limits."""
