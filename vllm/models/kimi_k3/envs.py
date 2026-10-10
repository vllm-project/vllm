# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3-specific environment variables.

Merged into :mod:`vllm.envs` at import time, so this module must only depend on
the standard library.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT: bool = False
    VLLM_KIMI_K3_AUX_ATTN_RES_STREAM: bool = False
    VLLM_KIMI_K3_GEMM_AR: bool = True

# --8<-- [start:env-vars-definition]
environment_variables: dict[str, Callable[[], Any]] = {
    # Under sequence-parallel MoE the dense and shared-expert MLPs are
    # replicated on every rank, so each rank streams the whole weight to serve
    # its own token shard. Shard them across TP instead: the MLP then
    # all-gathers the full token set, computes this rank's partial, and
    # reduce-scatters (which both sums across TP and restores the sequence
    # sharding). Trades weight bandwidth and resident memory for two collectives
    # per layer, so it only wins at low token counts: intended for decode
    # instances in a P/D disaggregated deployment, not for prefill or unified
    # serving.
    "VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT": lambda: bool(
        int(os.getenv("VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT", "0"))
    ),
    # Tap the pre-norm AttnRes mixture, rather than the post-mixture sum, as the
    # auxiliary hidden state handed to a DFlash drafter. This changes the
    # numerics the speculator sees, so it is off by default while the effect is
    # measured.
    "VLLM_KIMI_K3_AUX_ATTN_RES_STREAM": lambda: bool(
        int(os.getenv("VLLM_KIMI_K3_AUX_ATTN_RES_STREAM", "0"))
    ),
    # Use the SM100 BF16 GEMM-AR kernel for eligible row-parallel attention
    # projections. All TP ranks must belong to one NVLink domain.
    "VLLM_KIMI_K3_GEMM_AR": lambda: bool(int(os.getenv("VLLM_KIMI_K3_GEMM_AR", "1"))),
}
# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    # Delegate so reads share vllm.envs' cache.
    if name in environment_variables:
        import vllm.envs

        return getattr(vllm.envs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
