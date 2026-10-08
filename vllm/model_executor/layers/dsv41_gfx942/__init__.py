# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx942 (MI300X, MI325X) decode helpers for DeepSeek-V4.1.

``topk`` holds the sparse indexer's top-512 hooks, built from
``topk512_gfx942.cu``, and ``cand_logits`` the indexer logits of the DSpark
candidate positions. They and the model's other gfx942 decode helpers (the
top-k packing, the SWA decode metadata and DSpark's fused Markov sampling)
run only when ``enabled()`` is true.
"""

import functools

import vllm.envs as envs
from vllm.platforms import current_platform


@functools.cache
def enabled() -> bool:
    """True with VLLM_ROCM_MONO_DECODE=1 on a gfx942 GPU. Every helper keeps
    the result of vLLM's op it replaces, so each one returns False for a call
    it cannot take and vLLM then runs its own op. gfx950 always keeps vLLM's
    ops here."""
    if not envs.VLLM_ROCM_MONO_DECODE or not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942

    return on_gfx942()
