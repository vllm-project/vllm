# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in selection of Cake-generated FlashInfer kernels (``backend="cake"``).

``VLLM_CAKE_ROUTES`` is a comma-separated list of route names. With the variable
unset no Cake FlashInfer module is built or loaded and every call keeps its
existing backend. A route is admitted only when the installed FlashInfer offers
the Cake backend and the call matches the generated programs; the numerics of
an admitted call are FlashInfer's.

Routes:

* ``kda_decode`` -- Kimi-K3 fused KDA T=1 decode through
  ``flashinfer.kda_decode.fused_kda_decode(..., backend="cake")`` (by default
  this decode runs vLLM's own ``ops.fused_kda_decode`` CUDA op or Triton);
  decided once per layer (``vllm.models.kimi_k3.nvidia.kda``).
* ``kimi_k3_mla`` -- Kimi-K3 FP8 paged MLA T=1 decode through
  ``flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(..., backend="cake")``
  instead of FlashInfer's default backend; decided per decode-call shape
  (``vllm.v1.attention.backends.mla.flashinfer_mla``).
"""

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

KDA_DECODE_ROUTE = "kda_decode"
KIMI_K3_MLA_ROUTE = "kimi_k3_mla"
KNOWN_ROUTES: frozenset[str] = frozenset({KDA_DECODE_ROUTE, KIMI_K3_MLA_ROUTE})

# The Cake Kimi-K3 MLA programs reachable from vLLM's FlashInfer MLA decode:
# SM100 / SM103, FP8 e4m3 query and latent cache, kv_lora_rank 512 + rope 64
# (D = 576), page size 64, 12 query heads (TP8), one query token per request.
# The TP1 (96-head) and MTP (ragged) batches never reach the route in vLLM's
# wiring and are not admitted.
_KIMI_K3_MLA_CAPABILITIES = ((10, 0), (10, 3))
_KIMI_K3_MLA_HEADS = (12,)
_KIMI_K3_MLA_QK_DIM = 576
_KIMI_K3_MLA_PAGE_SIZE = 64


def cake_routes() -> frozenset[str]:
    """The route names selected by ``VLLM_CAKE_ROUTES`` (unknown names warn once)."""
    names = frozenset(s.strip() for s in envs.VLLM_CAKE_ROUTES.split(",") if s.strip())
    for name in sorted(names - KNOWN_ROUTES):
        logger.warning_once(
            "VLLM_CAKE_ROUTES names an unknown route %r (known: %s); ignored.",
            name,
            ", ".join(sorted(KNOWN_ROUTES)),
        )
    return names & KNOWN_ROUTES


def cake_route_enabled(name: str) -> bool:
    """Whether the Cake route ``name`` is selected by ``VLLM_CAKE_ROUTES``."""
    if name not in KNOWN_ROUTES:
        raise ValueError(f"unknown Cake route {name!r}")
    return name in cake_routes()


def kimi_k3_mla_decode_admits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    *,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    compute_capability: tuple[int, int] | None,
) -> bool:
    """Admission for the Cake Kimi-K3 FP8 paged MLA decode (host-side only).

    ``q`` is the uniform decode query ``[B, 1, H, D]``; ``kv_cache`` is the
    latent cache ``[pages, page_size, D]`` before the trtllm ``unsqueeze(1)``.
    """
    return (
        compute_capability in _KIMI_K3_MLA_CAPABILITIES
        and q.dtype == torch.float8_e4m3fn
        and kv_cache.dtype == torch.float8_e4m3fn
        and kv_lora_rank == 512
        and qk_rope_head_dim == 64
        and q.ndim == 4
        and int(q.shape[1]) == 1
        and int(q.shape[2]) in _KIMI_K3_MLA_HEADS
        and int(q.shape[3]) == _KIMI_K3_MLA_QK_DIM
        and kv_cache.ndim == 3
        and int(kv_cache.shape[1]) == _KIMI_K3_MLA_PAGE_SIZE
        and int(kv_cache.shape[2]) == _KIMI_K3_MLA_QK_DIM
    )
