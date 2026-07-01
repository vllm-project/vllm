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
    m_size_list = [1, 2, 4, 8, 16, 32, 64]
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
        (5120, 5120),
        (5120, 51200),
        (25600, 5120),
    ]
    has_bias_list = [False]

    in_dtype: torch.dtype = current_platform.fp8_dtype()
    scale_dtype: torch.dtype = torch.float32
    out_dtype: torch.dtype = torch.bfloat16
    inputs = {}
    for M, (K, N), has_bias in product(m_size_list, b_shape_list, has_bias_list):
        scale = 1.0 / math.sqrt(K)
        a = (scale * (0.5 + torch.rand(M, K, dtype=torch.float32, device="cuda"))).to(
            in_dtype
        )
        b = (scale * (0.5 + torch.rand(N, K, dtype=torch.float32, device="cuda"))).to(
            in_dtype
        )
        b = b.t()
        c = torch.empty((M, N), dtype=out_dtype, device=a.device)
        scale_a = 0.5 + torch.rand((1, M), dtype=scale_dtype, device="cuda")
        scale_a = scale_a.t()
        scale_b = 0.5 + torch.rand((1, 1), dtype=scale_dtype, device="cuda")
        bias = (
            0.5 * (torch.rand(N, dtype=out_dtype, device="cuda") - 0.5)
            if has_bias
            else None
        )

        config_key = CaseKey({"K": K, "N": N, "M": M, "bias": has_bias})
        inputs[config_key] = (c, a, b, scale_a, scale_b, bias)

    return inputs


_pick_cache: dict[tuple[int, int, int, bool], CaseKey | None] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape.

    K/N are picked by closest match. M is bucketed to the smallest tuned
    M >= runtime M. The bias field must match exactly.
    """

    if not config_keys:
        return None

    c, a, b, *_ = args
    bias = args[5] if len(args) > 5 else None

    M, K = a.shape
    N = b.shape[1]
    has_bias = bias is not None

    cache_key = (M, K, N, has_bias)
    if cache_key in _pick_cache:
        return _pick_cache[cache_key]

    configs: dict[int, dict[int, list[int]]] = {}
    for key in config_keys:
        if key.is_default():
            continue

        # Require exact bias match.
        if key["bias"] != has_bias:
            continue

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
            "bias": has_bias,
        }
    )
    _pick_cache[cache_key] = result
    return result


def fake_impl(
    c: torch.Tensor,  # [M, N]
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    scale_b: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
    bias: torch.Tensor | None = None,  # [N]
) -> None:
    return


def baseline(
    c: torch.Tensor,  # [M, N]
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    scale_b: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
    bias: torch.Tensor | None = None,  # [N]
) -> None:
    out = torch.mm(a.to(torch.float32), b.to(torch.float32))
    out = scale_a * out
    out = scale_b.T * out
    out = out.to(c.dtype)
    if bias is not None:
        out = out + bias

    c.copy_(out)


# Overwrite autotune_baseline_atol and autotune_baseline_rtol
# if too many configs failed due to baseline check during autotuning
@register_kernel(
    mutates_args=["c"],
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_baseline_fn=baseline,
        autotune_baseline_atol=1e-1,
        autotune_baseline_rtol=1e-1,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ),
)  # type: ignore[misc]
def scaled_mm(
    c: torch.Tensor,  # [M, N]
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    scale_b: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
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

    scale_a = scale_a.reshape(-1, 1) if scale_a.dim() <= 1 else scale_a
    scale_b = scale_b.reshape(-1, 1) if scale_b.dim() <= 1 else scale_b

    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()
    assert scale_a.shape[1] == 1 and (scale_a.shape[0] == 1 or scale_a.shape[0] == M)
    assert scale_b.shape[1] == 1 and (scale_b.shape[0] == 1 or scale_b.shape[0] == N)
    hl.specialize(scale_b.shape[1])

    out_dtype = c.dtype
    assert out_dtype.is_floating_point
    if bias is not None:
        assert bias.numel() == N and bias.dtype == out_dtype

    acc_dtype = torch.float32 if a.is_floating_point() else torch.int32

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], acc_dtype)
        for tile_k in hl.tile(K):
            acc = hl.dot(
                a[tile_m, tile_k],
                b[tile_k, tile_n],
                acc=acc,
                out_dtype=acc_dtype,
            )

        acc = acc.to(torch.float32)

        if scale_a.shape[0] == M:
            scale_a_blk = scale_a[tile_m, :]
        else:
            scale_a_blk = scale_a[0, 0].expand(tile_m.block_size, 1)
        acc = scale_a_blk * acc

        if scale_b.shape[0] == N:
            scale_b_blk = scale_b[tile_n, :]
        else:
            scale_b_blk = scale_b[0, 0].expand(tile_n.block_size, 1)
        acc = scale_b_blk.T * acc

        c_blk = acc.to(out_dtype)

        if bias is not None:
            c_blk += bias[tile_n]

        c[tile_m, tile_n] = c_blk
