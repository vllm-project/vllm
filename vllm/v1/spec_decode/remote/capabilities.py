# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capability model for dedicated-device speculator placement.

A speculator can only move to a dedicated GPU outside the target TP group
when all of its runtime state can be reconstructed from transported target
features (target-conditioned, own-KV). Compatibility is decided from an
explicit capability object reported by the standalone server for the actual
adapter and checkpoint, never from the speculative method name alone.
"""

import enum
from collections.abc import Collection
from typing import Literal

import msgspec

StateDependency = Literal["own_kv", "target_kv", "target_owned_state"]
StandaloneWeights = Literal["complete", "materializable", "missing"]

# Method families eligible for remote placement. This is a coarse
# config-time gate only: MTP variant names are normalized to "mtp" before
# validation, so variant- and checkpoint-level compatibility is enforced by
# the capability report in the HELLO handshake, not by this set.
REMOTE_DRAFT_SUPPORTED_METHODS = frozenset({"eagle", "eagle3", "mtp"})


class TargetFeatureKind(str, enum.Enum):
    """Target-side features a speculator may require every round."""

    TOKEN_IDS = "token_ids"
    POSITIONS = "positions"
    QUERY_LENS = "query_lens"
    LAST_HIDDEN_STATES = "last_hidden_states"
    AUX_HIDDEN_STATES = "aux_hidden_states"
    NEXT_PREFILL_TOKENS = "next_prefill_tokens"
    NUM_SAMPLED_TOKENS = "num_sampled_tokens"
    NUM_REJECTED_TOKENS = "num_rejected_tokens"
    TEMPERATURE = "temperature"
    SEEDS = "seeds"
    TARGET_TOPK_INDICES = "target_topk_indices"


class SpeculatorPlacementCapabilities(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    """What a concrete speculator adapter and checkpoint support.

    Reported by the standalone server during the HELLO handshake. The
    verifier compares this against its own configuration and the features it
    can provide, and rejects the connection with an explicit error instead
    of silently running an unsupported combination.

    Feature kinds are carried as plain strings (TargetFeatureKind values) so
    that a newer-minor peer advertising an unknown kind stays decodable and
    is rejected through `placement_incompatibilities` rather than failing
    with a codec error.
    """

    state_dependency: StateDependency
    required_features: tuple[str, ...]
    num_prefill_lookahead: int = 0
    """Lookahead tokens the adapter needs per prefill chunk; negotiation
    input for the prefill-chunk contract, not a reject gate."""
    supports_parallel_drafting: bool = False
    supports_multi_module: bool = False
    standalone_weights: StandaloneWeights = "missing"
    supports_probabilistic_draft: bool = False


def _feature_value(kind: str) -> str:
    return kind.value if isinstance(kind, TargetFeatureKind) else kind


def placement_incompatibilities(
    capabilities: SpeculatorPlacementCapabilities,
    *,
    parallel_drafting: bool,
    draft_sample_method: str,
    provided_features: Collection[str],
    uses_multi_module: bool = False,
) -> list[str]:
    """Return reasons why a negotiated placement is unsupported.

    An empty list means the verifier may use the remote speculator. Each
    entry is a human-readable reason suitable for a startup error message.
    ``provided_features`` accepts TargetFeatureKind members or their string
    values.
    """
    reasons: list[str] = []
    if capabilities.state_dependency != "own_kv":
        reasons.append(
            "the speculator depends on target-owned state "
            f"({capabilities.state_dependency!r}) that cannot be transported "
            "to a dedicated device; only own-KV speculators are supported"
        )
    if capabilities.standalone_weights == "missing":
        reasons.append(
            "the draft checkpoint does not provide the weights required to "
            "materialize the speculator in a standalone process (e.g. "
            "embedding or LM head shared with the target)"
        )
    provided = {_feature_value(f) for f in provided_features}
    missing = [
        _feature_value(f)
        for f in capabilities.required_features
        if _feature_value(f) not in provided
    ]
    if missing:
        reasons.append(
            f"the verifier cannot provide required target features: {sorted(missing)}"
        )
    if parallel_drafting and not capabilities.supports_parallel_drafting:
        reasons.append(
            "parallel_drafting is enabled but the speculator checkpoint does "
            "not support parallel drafting"
        )
    if uses_multi_module and not capabilities.supports_multi_module:
        reasons.append(
            "the draft checkpoint uses multiple speculative modules but the "
            "speculator does not support multi-module drafting"
        )
    if (
        draft_sample_method != "greedy"
        and not capabilities.supports_probabilistic_draft
    ):
        reasons.append(
            f"draft_sample_method={draft_sample_method!r} requires a draft "
            "probability contract that the remote speculator does not "
            "implement; only greedy draft sampling is supported"
        )
    return reasons


def remote_draft_config_incompatibilities(
    *,
    method: str | None,
    draft_tensor_parallel_size: int | None,
    draft_sample_method: str,
    uses_target_kv: bool | None,
) -> list[str]:
    """Return reasons why a SpeculativeConfig cannot use remote_draft.

    This is the config-time subset of the compatibility policy: everything
    that can be decided before connecting to the standalone server. The
    remaining checks (checkpoint capabilities, feature schema, fingerprints)
    happen during the HELLO handshake via `placement_incompatibilities`.
    ``uses_target_kv=None`` means the property could not be determined and
    is rejected fail-closed.
    """
    reasons: list[str] = []
    if method not in REMOTE_DRAFT_SUPPORTED_METHODS:
        reasons.append(
            f"method {method!r} is not supported with remote_draft; "
            f"supported methods: {sorted(REMOTE_DRAFT_SUPPORTED_METHODS)}"
        )
    if uses_target_kv is None:
        reasons.append(
            "cannot determine whether the draft model shares the target KV "
            "cache (draft model config is not resolved)"
        )
    elif uses_target_kv:
        reasons.append(
            "the speculator shares the target KV cache directly and cannot "
            "run on a dedicated device"
        )
    if draft_tensor_parallel_size is not None and draft_tensor_parallel_size > 1:
        reasons.append(
            "remote_draft requires draft_tensor_parallel_size=1; got "
            f"{draft_tensor_parallel_size}"
        )
    # Blanket config-time restriction; relax together with the
    # supports_probabilistic_draft gate in placement_incompatibilities.
    if draft_sample_method != "greedy":
        reasons.append(
            "remote_draft currently requires draft_sample_method='greedy'; "
            f"got {draft_sample_method!r}"
        )
    return reasons
