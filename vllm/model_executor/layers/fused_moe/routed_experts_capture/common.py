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

from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
)

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
