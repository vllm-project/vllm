# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer parallel configuration resolver.

Returns a ``LayerParallelConfig`` for shared layer code to query at
construction time. A default config (all fields ``None``) means "fall back
to the global TP world".
"""

from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class LayerParallelConfig:
    """Resolved per-layer parallel configuration.

    Fields default to ``None``, meaning the layer uses the global TP world.

    Attributes:
        tp_size: Effective TP world size this layer should shard with.
        tp_rank: This rank's index within ``tp_size``. Differs from the
            full TP rank when the attention TP group is smaller than the
            full TP group; QKV weight loading uses this rank.
    """

    tp_size: int | None = None
    tp_rank: int | None = None


_ATTENTION_PREFIX_PATTERNS: tuple[str, ...] = (
    ".self_attn.",
    ".attention.",
    ".attn.",
)


@dataclass(frozen=True)
class _ResolverState:
    full_tp_size: int
    full_tp_rank: int
    attn_tp_size: int
    attn_tp_rank: int


_resolver_state: _ResolverState | None = None


def init_layer_parallel_resolver(
    *,
    full_tp_size: int,
    full_tp_rank: int,
    attn_tp_size: int,
    attn_tp_rank: int,
) -> None:
    """Initialize the resolver with the engine's lowered parallel config.

    Called once from ``initialize_model_parallel`` after the TP and DCP
    groups are constructed. Subsequent calls reset the state.

    Args:
        full_tp_size: Full tensor-parallel world size.
        full_tp_rank: This worker's rank in the full TP group.
        attn_tp_size: Attention TP size. Must divide ``full_tp_size``.
        attn_tp_rank: This worker's rank within the attention TP group.
    """
    global _resolver_state
    if full_tp_size % attn_tp_size != 0:
        raise ValueError(
            "full_tp_size must be divisible by attn_tp_size: "
            f"got {full_tp_size=}, {attn_tp_size=}"
        )
    if not 0 <= attn_tp_rank < attn_tp_size:
        raise ValueError(f"attn_tp_rank out of range: {attn_tp_rank=}, {attn_tp_size=}")
    _resolver_state = _ResolverState(
        full_tp_size=full_tp_size,
        full_tp_rank=full_tp_rank,
        attn_tp_size=attn_tp_size,
        attn_tp_rank=attn_tp_rank,
    )
    if attn_tp_size != full_tp_size:
        logger.info(
            "Per-layer parallel resolver initialized: "
            "full_tp=%d, attn_tp=%d (TPA mode active)",
            full_tp_size,
            attn_tp_size,
        )


def clear_layer_parallel_resolver() -> None:
    """Reset the resolver state. Tests only."""
    global _resolver_state
    _resolver_state = None


def get_layer_parallel_config(
    layer: Any,  # noqa: ARG001
    prefix: str,
) -> LayerParallelConfig:
    """Resolve the per-layer parallel config for a layer at this prefix.

    Returns a config with non-``None`` fields when this layer differs from
    the global TP world (currently: attention layers under TPA). Otherwise
    callers fall back to ``get_tensor_model_parallel_world_size()`` /
    ``get_tensor_model_parallel_rank()``.
    """
    if _resolver_state is None:
        return LayerParallelConfig()

    state = _resolver_state
    if state.attn_tp_size == state.full_tp_size:
        return LayerParallelConfig()

    if not _is_attention_prefix(prefix):
        return LayerParallelConfig()

    return LayerParallelConfig(
        tp_size=state.attn_tp_size,
        tp_rank=state.attn_tp_rank,
    )


def _is_attention_prefix(prefix: str) -> bool:
    wrapped = f".{prefix}."
    return any(p in wrapped for p in _ATTENTION_PREFIX_PATTERNS)
