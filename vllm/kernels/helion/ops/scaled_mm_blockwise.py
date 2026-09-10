# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from itertools import product
from typing import Any

import torch

from vllm.kernels.helion.case_key import CaseKey
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion
import helion.language as hl

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


def generate_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    # TODO(xiaohongchen1991): it is difficult for kernel author to cover
    # all input property combination. Currently, dtypes are fixed. We need
    # optimization to bucket/skip some combinations
    m_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    b_shape_list = [
        # Qwen3-1.7B
        (2048, 4096),
        (2048, 2048),
        (2048, 12288),
        (6144, 2048),
        # Qwen3-8B
        (4096, 6144),
        (4096, 4096),
        (4096, 24576),
        (12288, 4096),
        # Qwen3-32B
        (5120, 10240),
        (5120, 5120),
        (5120, 51200),
        (25600, 5120),
    ]

    in_dtype: torch.dtype = current_platform.fp8_dtype()
    scale_dtype: torch.dtype = torch.float32
    out_dtype: torch.dtype = torch.bfloat16
    group_m = 1
    group_k = 128
    group_n = 128
    # generate_inputs is used for benchmarking purpose as well.
    # Currently, bias not yet supported by blockwise cutlass_scaled_mm.
    # Set it to None to avoid failure during benchmarking.
    bias = None
    inputs = {}
    for M, (K, N) in product(m_list, b_shape_list):
        scale = 1.0 / math.sqrt(K)
        a = (scale * (0.5 + torch.rand(M, K, dtype=torch.float32, device="cuda"))).to(
            in_dtype
        )
        b = (scale * (0.5 + torch.rand(N, K, dtype=torch.float32, device="cuda"))).to(
            in_dtype
        )
        b = b.t()
        num_group_m = M // group_m
        num_group_k = K // group_k
        num_group_n = N // group_n
        scale_a = 0.5 + torch.rand(
            num_group_m, num_group_k, dtype=scale_dtype, device="cuda"
        )
        scale_b = 0.5 + torch.rand(
            num_group_k, num_group_n, dtype=scale_dtype, device="cuda"
        )
        scale_a = scale_a.t().contiguous().t()
        scale_b = scale_b.t().contiguous().t()

        config_key = CaseKey(
            {
                "K": K,
                "N": N,
                "M": M,
            }
        )
        inputs[config_key] = (
            a,
            b,
            scale_a,
            scale_b,
            group_m,
            group_k,
            group_n,
            out_dtype,
            bias,
        )

    return inputs


_pick_cache: dict[tuple[int, int, int], CaseKey | None] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape.
    Selection strategy:
      1. Find the closest K among available configs
         (exact match preferred).
      2. Find the closest N among available configs
         (exact match preferred).
      3. Among the M values tuned for that K and N, pick
         the smallest M >= the input's M. If the input is
         larger than all available Ms, fall back to the largest.
    """

    if not config_keys:
        return None

    a, b, *_ = args
    M, K = a.shape
    N = b.shape[1]

    cache_key = (M, K, N)
    cached = _pick_cache.get(cache_key)
    if cached is not None:
        return cached

    configs: dict[int, dict[int, list[int]]] = {}
    for key in config_keys:
        if key.is_default():
            continue
        configs.setdefault(key["K"], {}).setdefault(key["N"], []).append(key["M"])

    if not configs:
        return None

    best_K = min(configs, key=lambda s: abs(s - K))
    best_N = min(configs[best_K], key=lambda s: abs(s - N))
    available_M = sorted(configs[best_K][best_N])
    best_M = next((m for m in available_M if m >= M), available_M[-1])

    result = CaseKey(
        {
            "K": best_K,
            "N": best_N,
            "M": best_M,
        }
    )
    _pick_cache[cache_key] = result
    return result


def fake_impl(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [num_group_m, num_group_k]
    scale_b: torch.Tensor,  # [num_group_k, num_group_n]
    group_m: int,
    group_k: int,
    group_n: int,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,  # [N]
) -> torch.Tensor:
    M = a.shape[0]
    N = b.shape[1]
    c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    return c


def baseline(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [num_group_m, num_group_k]
    scale_b: torch.Tensor,  # [num_group_k, num_group_n]
    group_m: int,
    group_k: int,
    group_n: int,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,  # [N]
) -> torch.Tensor:
    def group_broadcast(t, shape):
        for i, s in enumerate(shape):
            if t.shape[i] != s and t.shape[i] != 1:
                assert s % t.shape[i] == 0
                t = (
                    t.unsqueeze(i + 1)
                    .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                    .flatten(i, i + 1)
                )
        return t

    scale_a = group_broadcast(scale_a, a.shape)
    scale_b = group_broadcast(scale_b, b.shape)

    out = torch.mm(
        (scale_a * a.to(dtype=torch.float32)), (scale_b * b.to(dtype=torch.float32))
    ).to(out_dtype)

    if bias is not None:
        out = out + bias

    return out


# Overwrite autotune_baseline_atol and autotune_baseline_rtol
# if too many configs failed due to baseline check during autotuning
@register_kernel(
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_baseline_fn=baseline,
        autotune_baseline_atol=1.0,
        autotune_baseline_rtol=5e-1,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ),
)  # type: ignore[misc]
def scaled_mm_blockwise(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [num_group_m, num_group_k]
    scale_b: torch.Tensor,  # [num_group_k, num_group_n]
    group_m: int,
    group_k: int,
    group_n: int,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,  # [N]
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    M, K = a.shape
    N = b.shape[1]
    hl.specialize(K)
    hl.specialize(N)

    assert N > 0 and K > 0 and M > 0
    assert b.shape[0] == K
    assert a.dtype == b.dtype

    assert a.stride(1) == 1
    assert b.stride(0) == 1

    assert bias is None

    assert scale_a.ndim == 2 and scale_b.ndim == 2
    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()

    num_group_m, num_group_k = scale_a.shape
    num_group_n = scale_b.shape[1]
    hl.specialize(num_group_k)
    hl.specialize(num_group_n)

    assert M % num_group_m == 0
    assert K % num_group_k == 0
    assert N % num_group_n == 0

    # scale_a group shape must be [1, 128]
    # scale_b group shape must be [128, 128]
    assert scale_b.shape[0] == num_group_k
    assert M // num_group_m == group_m
    assert K // num_group_k == group_k
    assert N // num_group_n == group_n
    assert group_m == 1 and group_k == 128 and group_n == 128

    hl.specialize(group_m)
    hl.specialize(group_k)
    hl.specialize(group_n)

    assert out_dtype.is_floating_point

    c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    acc_dtype = torch.float32 if a.is_floating_point() else torch.int32

    for tile_m, tile_n in hl.tile([M, N]):
        accumulator = hl.zeros(
            [tile_m, tile_n],
            torch.float32,
        )

        # keep block_size = group_k for K dimension to avoid element-wise scaling
        for tile_k in hl.tile(K, block_size=group_k):
            a_blk = a[tile_m, tile_k]
            b_blk = b[tile_k, tile_n]

            acc_blk = hl.dot(
                a_blk,
                b_blk,
                out_dtype=acc_dtype,
            ).to(torch.float32)

            gk_idx = tile_k.begin // group_k
            scale_a_blk = scale_a[tile_m, gk_idx][:, None]
            scale_b_blk = scale_b[gk_idx, tile_n.index // group_n][None, :]

            acc_blk = scale_a_blk * acc_blk
            acc_blk = scale_b_blk * acc_blk

            accumulator = accumulator + acc_blk

        c_blk = accumulator.to(out_dtype)

        c[tile_m, tile_n] = c_blk

    return c
