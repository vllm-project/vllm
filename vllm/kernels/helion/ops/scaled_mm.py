# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 RowWise ``scaled_mm`` Helion kernel (per-token act / per-channel weight).

    out[m, n] = scale_a[m] * scale_b[n] * sum_k A[m, k] * B[k, n]

A/B fp8 e4m3 (B column-major ``[K, N]`` as vLLM stores weights), out bf16,
scale_a ``[M, 1]`` f32, scale_b ``[1, N]`` f32.

This is the skinny-M (decode) regime: memory-bandwidth bound on reading B, so a
split-K reduction (``split_k`` tunable) is used to fill the machine, with the
rowwise scale folded into each K-split partial (linear across K) and partials
accumulated via ``atomic_add`` into a pre-zeroed output. Beats / matches cutlass
at M<=16; the caller routes only small M here (see the FP8 linear kernel).

N and K are specialized (fixed per layer); M (num tokens) stays dynamic.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.kernels.helion.case_key import CaseKey
from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "scaled_mm Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion
import helion.language as hl
from helion.autotuner import PowerOfTwoFragment

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)

# (N, K) of the FP8 W8A8 linear layers we tune for (Qwen3-1.7B); N=out, K=in.
_LAYER_NK = [(4096, 2048), (2048, 2048), (12288, 2048), (2048, 6144)]
# Decode num_tokens (M) where the split-K kernel matches/beats cutlass.
_NUM_TOKENS = [1, 2, 4, 8, 16]


def generate_scaled_mm_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    inputs: dict[CaseKey, tuple[Any, ...]] = {}
    for n, k in _LAYER_NK:
        for m in _NUM_TOKENS:
            a = (torch.randn(m, k, device="cuda") * 0.3).to(torch.float8_e4m3fn)
            # Column-major [K, N], matching how vLLM stores the fp8 weight.
            b = (
                (torch.randn(k, n, device="cuda") * 0.3)
                .to(torch.float8_e4m3fn)
                .T.contiguous()
                .T
            )
            scale_a = (torch.rand(m, 1, device="cuda") + 0.5).to(torch.float32)
            scale_b = (torch.rand(1, n, device="cuda") + 0.5).to(torch.float32)
            inputs[CaseKey({"n": n, "k": k, "numtokens": m})] = (a, b, scale_a, scale_b)
    return inputs


_pick_cache: dict[tuple[int, int, int], CaseKey | None] = {}


def pick_scaled_mm_config(
    args: tuple[Any, ...], config_keys: list[CaseKey]
) -> CaseKey | None:
    """Pick the pre-tuned config: exact (n, k) match, then smallest tuned
    num_tokens >= the input's M (fall back to the largest tuned M)."""
    if not config_keys:
        return None

    a, b, _sa, _sb = args
    m = int(a.shape[0])
    k = int(a.shape[1])
    n = int(b.shape[1])

    cache_key = (m, n, k)
    cached = _pick_cache.get(cache_key)
    if cached is not None:
        return cached

    by_nk: dict[tuple[int, int], list[int]] = {}
    for key in config_keys:
        if key.is_default():
            continue
        by_nk.setdefault((key["n"], key["k"]), []).append(key["numtokens"])

    if (n, k) not in by_nk:
        return None
    available = sorted(by_nk[(n, k)])
    best_m = next((x for x in available if x >= m), available[-1])
    result = CaseKey({"n": n, "k": k, "numtokens": best_m})
    _pick_cache[cache_key] = result
    return result


def _scaled_mm_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((A.shape[0], B.shape[1]), dtype=torch.bfloat16, device=A.device)


@register_kernel(
    config_picker=pick_scaled_mm_config,
    input_generator=generate_scaled_mm_inputs,
    fake_impl=_scaled_mm_fake,
)
def scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    m, k = A.shape
    _, n = B.shape
    n = hl.specialize(n)
    k = hl.specialize(k)
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 32))
    out = torch.zeros([m, n], dtype=torch.bfloat16, device=A.device)
    k_block = helion.next_power_of_2(helion.cdiv(k, split_k))
    for tile_m, tile_n, outer_k in hl.tile([m, n, k], block_size=[None, None, k_block]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for inner_k in hl.tile(outer_k.begin, outer_k.end):
            acc = hl.dot(A[tile_m, inner_k], B[inner_k, tile_n], acc=acc)
        # Rowwise scale is linear across K -> fold into each split-K partial.
        acc = acc * scale_a[tile_m, :] * scale_b[:, tile_n]
        hl.atomic_add(out, [tile_m, tile_n], acc.to(torch.bfloat16))
    return out


def scaled_mm_baseline(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    from vllm import _custom_ops as ops

    return ops.cutlass_scaled_mm(
        A, B, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16
    )
