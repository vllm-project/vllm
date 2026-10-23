# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx942 (MI300X, MI325X) decode helpers for DeepSeek-V4.1.

``topk`` holds the sparse indexer's top-512 hooks, built from
``topk512_gfx942.cu``, and ``cand_logits`` the indexer logits of the DSpark
candidate positions. ``sparse_indexer`` is the indexer layer that runs them on
decode steps. ``topk_pack`` and ``swa_decode_meta`` build the decode
attention metadata, ``fused_markov`` is DSpark's Markov sampling in one
launch, and ``draft_lm_head`` keeps the DSpark draft's 5-row logits GEMM off
wvSplitK. All of them run only when ``enabled()`` is true.
"""

import functools

from vllm.platforms import current_platform

# --monokernel (KernelConfig.specialized_decode) of this process's model. The
# helpers run in forward passes, where vLLM's current config is not set, so the
# model records the flag when it is created (record_specialized_decode).
_specialized_decode = False


def record_specialized_decode(vllm_config) -> None:
    """Record whether this process's model runs with --monokernel. The
    specialized decode records it when the model is created, before any helper
    runs."""
    global _specialized_decode
    _specialized_decode = bool(vllm_config.kernel_config.specialized_decode)
    enabled.cache_clear()


@functools.cache
def enabled() -> bool:
    """True with --monokernel on a gfx942 GPU. Every helper keeps the result
    of vLLM's op it replaces, so each one returns False for a call it cannot
    take and vLLM then runs its own op. gfx950 always keeps vLLM's ops here."""
    if not _specialized_decode or not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942

    return on_gfx942()
