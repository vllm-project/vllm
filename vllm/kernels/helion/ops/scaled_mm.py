# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


def generate_scaled_mm_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    """Generate Qwen3-1.7B shapes: 4 layer GEMMs × M ∈ {1,2,4,8,16,32,64,512}."""
    # Qwen3-1.7B layer dimensions (K, N) - typical small LLM shapes
    # These correspond to QKV proj, O proj, gate/up proj, down proj
    layer_shapes = [
        (1536, 1536),  # QKV projection
        (1536, 1536),  # O projection
        (1536, 8960),  # gate/up projection (intermediate)
        (8960, 1536),  # down projection
    ]

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 512]

    inputs: dict[CaseKey, tuple[Any, ...]] = {}
    for m in batch_sizes:
        for k, n in layer_shapes:
            a = torch.randn(m, k, device="cuda", dtype=torch.float32).to(
                torch.float8_e4m3fn
            )
            b = torch.randn(k, n, device="cuda", dtype=torch.float32).to(
                torch.float8_e4m3fn
            )
            scale_a = torch.rand(m, 1, device="cuda", dtype=torch.float32) * 0.25 + 0.875
            scale_b = torch.rand(1, n, device="cuda", dtype=torch.float32) * 0.25 + 0.875

            key = CaseKey({"m": m, "k": k, "n": n})
            inputs[key] = (a, b, scale_a, scale_b)

    return inputs


_pick_cache: dict[tuple[int, int, int], CaseKey | None] = {}


def pick_scaled_mm_config(
    args: tuple[Any, ...], config_keys: list[CaseKey]
) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Try exact match on (m, k, n)
      2. If no exact match, find closest k, then closest n, then closest m
    """
    if not config_keys:
        return None

    a, b, _scale_a, _scale_b = args
    m, k = a.shape
    _, n = b.shape

    cache_key = (m, k, n)
    cached = _pick_cache.get(cache_key)
    if cached is not None:
        return cached

    # Try exact match first
    for key in config_keys:
        if not key.is_default() and key["m"] == m and key["k"] == k and key["n"] == n:
            _pick_cache[cache_key] = key
            return key

    # Find closest match: prioritize K dimension (most critical for GEMM)
    by_k: dict[int, list[tuple[int, int, CaseKey]]] = {}
    for key in config_keys:
        if key.is_default():
            continue
        by_k.setdefault(key["k"], []).append((key["m"], key["n"], key))

    if not by_k:
        return None

    # Find closest K
    best_k = min(by_k, key=lambda x: abs(x - k))
    candidates = by_k[best_k]

    # Among those, find closest N
    by_n: dict[int, list[tuple[int, CaseKey]]] = {}
    for m_val, n_val, key in candidates:
        by_n.setdefault(n_val, []).append((m_val, key))

    best_n = min(by_n, key=lambda x: abs(x - n))
    final_candidates = by_n[best_n]

    # Among those, find closest M
    best_m, result = min(final_candidates, key=lambda x: abs(x[0] - m))
    _pick_cache[cache_key] = result
    return result


@register_kernel(
    config_picker=pick_scaled_mm_config,
    input_generator=generate_scaled_mm_inputs,
    helion_settings=helion.Settings(backend="cute"),
)
def scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 RowWise scaled matrix multiplication on Helion's CuTe (tcgen05) backend.

    Computes: out[m,n] = scale_a[m] * scale_b[n] * sum_k(a[m,k] * b[k,n])

    The rowwise scale is fused in the epilogue. ``scale_a`` (per-row) is read as
    a stride-(1,0) ``(M, N)`` view so the backend reads it as a scalar per
    subtile; ``scale_b`` (per-column) is a rank-1 row-vector register-hoisted
    before the accumulator wait. This mirrors the tuned helion
    ``examples/scaled_mm.py`` kernel and the deep-pipeline 2-CTA config carried
    in the B200 preset. ``b`` may be row- or column-major: the CuTe backend
    handles both operand-major modes (CUTLASS passes column-major ``[K, N]``).

    Args:
        a: Input matrix [M, K] in FP8 e4m3 (row-major).
        b: Weight matrix [K, N] in FP8 e4m3.
        scale_a: Row-wise scale factors [M, 1] in FP32.
        scale_b: Column-wise scale factors [1, N] in FP32.

    Returns:
        Output matrix [M, N] in BF16.
    """
    m, k = a.size()
    _, n = b.size()
    sa2d = scale_a.reshape(m, 1).expand(m, n)
    sb1d = scale_b.reshape(n)
    out = torch.empty([m, n], dtype=torch.bfloat16, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = (acc * sa2d[tile_m, tile_n] * sb1d[tile_n]).to(
            torch.bfloat16
        )
    return out


def scaled_mm_baseline(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """Baseline implementation using cutlass scaled_mm."""
    import vllm._custom_ops as ops
    return ops.cutlass_scaled_mm(
        a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16
    )
