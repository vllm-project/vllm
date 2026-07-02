# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared, torch-free helpers for routed-experts capture/offload.

Both the worker-side capturer and the scheduler-side manager must agree on
which KV cache group anchors the routed-experts slot mapping, and on how the
MoE expert counts are read from the HF config. Keeping these here (no torch,
no numpy) lets the worker import the capturer without pulling in the scheduler
manager, and vice versa.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
)

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

logger = init_logger(__name__)

# Spec kinds whose per-token slot layout spans the whole sequence — the
# prerequisite for anchoring routed-experts slot mapping. MLA is a latent
# full-attention variant; sliding-window / chunked-local / Mamba are not.
_FULL_ATTENTION_KINDS = frozenset(
    {
        KVCacheSpecKind.FULL_ATTENTION,
        KVCacheSpecKind.MLA_ATTENTION,
        KVCacheSpecKind.SINK_FULL_ATTENTION,
    }
)


def find_full_attention_gid(kv_cache_config: KVCacheConfig) -> int | None:
    """Attention-slot anchor selection rule for routed-experts capture.

    Both the scheduler-side RoutedExpertsManager and the worker-side
    GPUModelRunner must derive slot mappings from the SAME KV cache
    group; sharing this helper is a correctness requirement, not a
    convenience. ``get_kv_cache_spec_kind`` unwraps the
    ``UniformTypeKVCacheSpecs`` groups used by DSA / DeepSeek-V4 (which
    wrap same-type MLA layers), so those anchor on their inner kind.
    Returns None when the model has no full-attention group (pure
    sliding-window / Mamba models are unsupported).
    """
    for gid, g in enumerate(kv_cache_config.kv_cache_groups):
        if get_kv_cache_spec_kind(g.kv_cache_spec) in _FULL_ATTENTION_KINDS:
            return gid
    return None


def require_full_attention_gid(kv_cache_config: KVCacheConfig) -> int:
    """Resolve the routed-experts anchor group id, raising if none exists.

    The single gate every routed-experts caller (scheduler, worker model
    runner, manager) goes through, so the "needs a full-attention group" rule
    and the multi-group policy live here instead of being re-checked at each
    site. Routing anchors on the FIRST full-attention group; a hybrid model
    with more than one warns, since routing for tokens whose KV lives in the
    other group(s) is not offloaded.

    Raises:
        ValueError: The model has no full-attention KV group (pure
            sliding-window / Mamba models are unsupported).
    """
    full_attn_gids = [
        gid
        for gid, g in enumerate(kv_cache_config.kv_cache_groups)
        if get_kv_cache_spec_kind(g.kv_cache_spec) in _FULL_ATTENTION_KINDS
    ]
    if not full_attn_gids:
        raise ValueError(
            "enable_return_routed_experts requires at least one full-attention "
            "KV cache group; pure sliding-window / Mamba models are unsupported."
        )
    if len(full_attn_gids) > 1:
        logger.warning_once(
            "enable_return_routed_experts: %d full-attention KV cache groups "
            "%s; anchoring routing on group %d only. Routing for tokens whose "
            "KV lives in the other group(s) is not offloaded.",
            len(full_attn_gids),
            full_attn_gids,
            full_attn_gids[0],
        )
    return full_attn_gids[0]


def get_num_experts_per_tok(hf_config) -> int:
    """Resolve the per-token expert count from the HF config.

    Different model families store this under different attribute names
    (e.g. ``num_experts_per_tok`` for DeepSeek, ``top_k_experts`` for Gemma 4).
    """
    val = getattr(hf_config, "num_experts_per_tok", None)
    if val is None:
        val = getattr(hf_config, "top_k_experts", None)
    if val is None:
        raise ValueError(
            "Cannot determine num_experts_per_tok: HF config has neither "
            "'num_experts_per_tok' nor 'top_k_experts'"
        )
    return val


def get_num_experts(hf_config) -> int:
    """Resolve ``num_experts`` across HuggingFace config naming conventions.

    Different MoE model families expose this under different keys:
      - ``num_experts``: Mixtral, Qwen2-MoE, Qwen3-MoE
      - ``n_routed_experts``: DeepSeek-V2/V3
      - ``num_local_experts``: Mixtral (older exports)

    The returned value must be the GLOBAL logical expert count: it sizes
    the dtype of the scheduler-side slot buffer, and captured IDs are
    logical (taken before the EPLB logical->physical mapping). All keys
    above are checkpoint-level globals -- ``num_local_experts`` is
    Mixtral's HF name for the total expert count, NOT a per-EP-rank
    count; vLLM's EP sharding never rewrites the HF config. Do not
    replace this with a per-rank source.
    """
    for key in ("num_experts", "n_routed_experts", "num_local_experts"):
        val = getattr(hf_config, key, None)
        if val is not None:
            return val
    raise ValueError(
        "Could not resolve num_experts from model config. "
        "Expected one of 'num_experts', 'n_routed_experts', "
        "or 'num_local_experts'."
    )


def routing_slot_shape_dtype(
    vllm_config, kv_cache_config: KVCacheConfig
) -> tuple[tuple[int, int, int], str]:
    """Slot-buffer shape ``(num_blocks*block_size, layers, top_k)`` + dtype.

    Single source of truth so the scheduler (``RoutedExpertsManager``) and the
    worker writer derive an IDENTICAL shared-buffer layout — they must agree
    exactly or the shared mmap mismatches. dtype is the narrow expert-id type
    (``uint8`` if ``num_experts <= 256`` else ``uint16``). Mirrors the sizing
    inside ``RoutedExpertsManager.__init__``.
    """

    attn_gid = require_full_attention_gid(kv_cache_config)
    block_size = kv_cache_config.kv_cache_groups[attn_gid].kv_cache_spec.block_size
    hf_config = vllm_config.model_config.hf_text_config
    num_layers = hf_config.num_hidden_layers
    top_k = get_num_experts_per_tok(hf_config)
    num_experts = get_num_experts(hf_config)
    max_num_slots = kv_cache_config.num_blocks * block_size
    dtype = "uint8" if num_experts <= 256 else "uint16"
    return (max_num_slots, num_layers, top_k), dtype


def routed_experts_output_rank(parallel_config: ParallelConfig) -> int:
    """Global rank whose ``ModelRunnerOutput`` reaches the scheduler.

    That rank is the sole writer of the shared routing slot buffer. Both the
    worker model runner and the Mooncake bridge resolve it here so they never
    disagree on which rank writes routing.

    The formula ``world_size - tp_size * pcp_size`` only points at the
    scheduler-adjacent rank when PP == 1. PP > 1 is rejected at config time
    (``config/vllm.py``), but assert it here too so the invariant travels with
    the formula: if that guard is ever relaxed, this fails loud instead of
    silently selecting a rank whose output never reaches the scheduler (which
    would yield all-``None`` routing).
    """
    pc = parallel_config
    assert pc.pipeline_parallel_size == 1, (
        "routed_experts_output_rank assumes PP == 1 (enforced in config); "
        f"got pipeline_parallel_size={pc.pipeline_parallel_size}"
    )
    return pc.world_size - pc.tensor_parallel_size * pc.prefill_context_parallel_size
