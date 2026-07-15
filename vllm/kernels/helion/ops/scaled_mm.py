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
    # The Helion linear kernel is autotuned per shape.
    # m_size_list follows cudagraph_capture_sizes pattern:
    # [1, 2, 4] + range(8, 256, 8) + range(256, max_graph_size + 1, 16),
    # but is capped here to cover only small M values.
    m_size_list = [1, 2, 4, 8, 16, 24, 32]
    b_shape_list = [
        # qwen3-1.7B
        # TP=1
        (2048, 4096),
        (2048, 2048),
        (2048, 12288),
        (6144, 2048),
        # qwen3-8B
        # TP=1
        (4096, 6144),
        (4096, 4096),
        (4096, 24576),
        (12288, 4096),
        # qwen3-32B
        # TP=1
        (5120, 10240),
        (8192, 5120),
        (5120, 51200),
        (25600, 5120),
    ]
    has_bias = False

    in_dtype: torch.dtype = current_platform.fp8_dtype()
    scale_dtype: torch.dtype = torch.float32
    out_dtype: torch.dtype = torch.bfloat16
    inputs = {}
    for M, (K, N) in product(m_size_list, b_shape_list):
        scale = 1.0 / math.sqrt(K)
        a = (scale * (0.5 + torch.rand(M, K, dtype=torch.float32, device="cuda"))).to(
            in_dtype
        )
        b = (scale * (0.5 + torch.rand(N, K, dtype=torch.float32, device="cuda"))).to(
            in_dtype
        )
        b = b.t()
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        a_scales = 0.5 + torch.rand((1, M), dtype=scale_dtype, device="cuda")
        a_scales = a_scales.t()
        b_scales = 0.5 + torch.rand((N, 1), dtype=scale_dtype, device="cuda")
        bias = (
            0.5 * (torch.rand(N, dtype=out_dtype, device="cuda") - 0.5)
            if has_bias
            else None
        )

        config_key = CaseKey({"K": K, "N": N, "M": M})
        inputs[config_key] = (out, a, b, a_scales, b_scales, bias)

    return inputs


_pick_cache: dict[tuple[int, int, int], CaseKey | None] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape.

    K/N are picked by closest match. M is bucketed to the smallest tuned
    M >= runtime M.
    """

    if not config_keys:
        return None

    out, a, b, *_ = args

    M, K = a.shape
    N = b.shape[1]

    cache_key = (M, K, N)
    if cache_key in _pick_cache:
        return _pick_cache[cache_key]

    configs: dict[int, dict[int, list[int]]] = {}
    for key in config_keys:
        if key.is_default():
            continue

        if all(k in key for k in ("K", "N", "M")):
            configs.setdefault(key["K"], {}).setdefault(key["N"], []).append(key["M"])

    if not configs:
        _pick_cache[cache_key] = None
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
    out: torch.Tensor,  # [M, N]
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    a_scales: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    b_scales: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
    bias: torch.Tensor | None = None,  # [N]
) -> None:
    return


def baseline(
    out: torch.Tensor,  # [M, N]
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    a_scales: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    b_scales: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
    bias: torch.Tensor | None = None,  # [N]
) -> None:
    c = torch.mm(a.to(torch.float32), b.to(torch.float32))
    c = a_scales * c
    c = b_scales.T * c
    c = c.to(out.dtype)
    if bias is not None:
        c = c + bias

    out.copy_(c)


# TODO(xiaohongchen1991):
# 1. Remove ProcessGroupNameNotFound from ignore_warning after fix for
# https://github.com/pytorch/helion/issues/3024 is available in vLLM
# 2. Conditionally use SwapAB trick when the fix for
# https://github.com/pytorch/helion/issues/3044 is available in vLLM


# Quantized GEMM kernels can have relatively large numerical differences
# from the baseline.
# Override autotune_baseline_atol and autotune_baseline_rtol to prevent
# excessive config failures from baseline accuracy checks during autotuning.
@register_kernel(
    mutates_args=["out"],
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_baseline_fn=baseline,
        autotune_baseline_atol=1e-1,
        autotune_baseline_rtol=1e-1,
        ignore_warnings=[
            helion.exc.TensorOperationInWrapper,
            helion.exc.ProcessGroupNameNotFound,
        ],
    ),
)  # type: ignore[misc]
def scaled_mm(
    out: torch.Tensor,  # [M, N]
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    a_scales: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    b_scales: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
    bias: torch.Tensor | None = None,  # [N]
) -> None:
    M, K = a.shape
    N = b.shape[1]
    hl.specialize(K)
    hl.specialize(N)

    assert N > 0 and K > 0 and M > 0
    assert b.shape[0] == K
    assert a.dtype == b.dtype
    assert a.stride(1) == 1
    assert b.stride(0) == 1

    a_scales = a_scales.reshape(-1, 1) if a_scales.dim() <= 1 else a_scales
    b_scales = b_scales.reshape(-1, 1) if b_scales.dim() <= 1 else b_scales

    assert a_scales.dtype == b_scales.dtype and a_scales.is_floating_point()
    assert a_scales.shape[1] == 1 and (a_scales.shape[0] == 1 or a_scales.shape[0] == M)
    assert b_scales.shape[1] == 1 and (b_scales.shape[0] == 1 or b_scales.shape[0] == N)
    hl.specialize(b_scales.shape[1])

    out_dtype = out.dtype
    assert out_dtype.is_floating_point
    if bias is not None:
        assert bias.numel() == N and bias.dtype == out_dtype

    acc_dtype = torch.float32 if a.is_floating_point() else torch.int32

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_n, tile_m], acc_dtype)
        for tile_k in hl.tile(K):
            a_blk = hl.load(a, [tile_m.index[None, :], tile_k.index[:, None]])
            b_blk = hl.load(b, [tile_k.index[None, :], tile_n.index[:, None]])
            acc = hl.dot(
                b_blk,
                a_blk,
                acc=acc,
                out_dtype=acc_dtype,
            )

        acc = acc.t().to(torch.float32)

        if a_scales.shape[0] == M:
            a_scales_blk = a_scales[tile_m, :]
        else:
            a_scales_blk = a_scales[0, 0].expand(tile_m.block_size, 1)
        acc = a_scales_blk * acc

        if b_scales.shape[0] == N:
            b_scales_blk = b_scales[tile_n, :]
        else:
            b_scales_blk = b_scales[0, 0].expand(tile_n.block_size, 1)
        acc = b_scales_blk.T * acc

        out_blk = acc.to(out_dtype)

        if bias is not None:
            out_blk += bias[tile_n]

        out[tile_m, tile_n] = out_blk
