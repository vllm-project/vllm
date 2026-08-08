# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from itertools import product
from typing import Any

import torch

from vllm.kernels.helion.case_key import CaseKey
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import _ceil_to_ue8m0
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion
import helion.language as hl
from helion.autotuner import PowerOfTwoFragment

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


def generate_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    # TODO(xiaohongchen1991): it is difficult for kernel author to cover
    # all input property combination. Currently, dtypes are fixed. We need
    # optimization to bucket/skip some combinations
    m_size_list = [1, 2, 4, 8, 16, 24, 32]

    fp8: torch.dtype = current_platform.fp8_dtype()

    b_shape_dtype_list: list[tuple[tuple[int, int], torch.dtype]] = [
        # qwen3-1.7B
        # TP=1
        ((2048, 4096), fp8),
        ((2048, 2048), fp8),
        ((2048, 12288), fp8),
        ((6144, 2048), fp8),
        # qwen3-4B
        # TP=1
        ((2560, 6144), fp8),
        ((4096, 2560), fp8),
        ((2560, 19456), fp8),
        ((9728, 2560), fp8),
        # qwen3-8B
        # TP=1
        ((4096, 6144), fp8),
        ((4096, 4096), fp8),
        ((4096, 24576), fp8),
        ((12288, 4096), fp8),
        # qwen3-14B
        # TP=1
        ((5120, 7168), fp8),
        ((5120, 5120), fp8),
        ((5120, 34816), fp8),
        ((17408, 5120), fp8),
        # qwen3-32B
        # TP=1
        ((5120, 10240), fp8),
        ((8192, 5120), fp8),
        ((5120, 51200), fp8),
        ((25600, 5120), fp8),
    ]

    scale_dtype: torch.dtype = torch.float32
    out_dtype: torch.dtype = torch.bfloat16
    group_m = 1
    group_k = 128
    group_n = 128

    inputs = {}
    for M, ((K, N), in_dtype) in product(m_size_list, b_shape_dtype_list):
        scale = 1.0 / math.sqrt(K)
        if in_dtype.is_floating_point:
            a = (
                scale * (0.5 + torch.rand(M, K, dtype=torch.float32, device="cuda"))
            ).to(in_dtype)
            b = (
                scale * (0.5 + torch.rand(N, K, dtype=torch.float32, device="cuda"))
            ).to(in_dtype)
        else:
            a = torch.randint(-32, 32, (M, K), dtype=in_dtype, device="cuda")
            b = torch.randint(-32, 32, (N, K), dtype=in_dtype, device="cuda")
        b = b.t()
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        num_group_m = M // group_m
        num_group_k = K // group_k
        num_group_n = N // group_n
        a_scales = 0.5 + torch.rand(
            num_group_m, num_group_k, dtype=scale_dtype, device="cuda"
        )
        b_scales = 0.5 + torch.rand(
            num_group_n, num_group_k, dtype=scale_dtype, device="cuda"
        )
        b_scales = b_scales.t()
        # Scales arrive in FLOAT32_CEIL_UE8M0 format (fp32 values ceiled to
        # powers of two) from the DeepGEMM e8m0 quant path on Hopper/Blackwell.
        a_scales = _ceil_to_ue8m0(a_scales)
        b_scales = _ceil_to_ue8m0(b_scales)

        config_key = CaseKey(
            {
                "K": K,
                "N": N,
                "M": M,
                "in_dtype": str(in_dtype),
            }
        )
        inputs[config_key] = (
            out,
            a,
            b,
            a_scales,
            b_scales,
        )

    return inputs


_pick_cache: dict[tuple[int, int, int, str], CaseKey | None] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape.

    Configs are matched within the runtime input dtype. K/N are picked by
    closest match. M is bucketed to the smallest tuned M >= runtime M.
    """

    if not config_keys:
        return None

    out, a, b, *_ = args

    M, K = a.shape
    N = b.shape[1]
    in_dtype = str(a.dtype)

    cache_key = (M, K, N, in_dtype)
    if cache_key in _pick_cache:
        return _pick_cache[cache_key]

    configs: dict[int, dict[int, list[int]]] = {}
    for key in config_keys:
        if key.is_default():
            continue

        if all(k in key for k in ("K", "N", "M", "in_dtype")):
            if "in_dtype" not in key or key["in_dtype"] != in_dtype:
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
            "in_dtype": in_dtype,
        }
    )
    _pick_cache[cache_key] = result
    return result


def fake_impl(
    out: torch.Tensor,  # [M, N]
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    a_scales: torch.Tensor,  # [num_group_m, num_group_k]
    b_scales: torch.Tensor,  # [num_group_k, num_group_n]
) -> None:
    return


def baseline(
    out: torch.Tensor,  # [M, N]
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    a_scales: torch.Tensor,  # [num_group_m, num_group_k]
    b_scales: torch.Tensor,  # [num_group_k, num_group_n]
) -> None:
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

    a_scales = group_broadcast(a_scales, a.shape)
    b_scales = group_broadcast(b_scales, b.shape)

    c = torch.mm(
        (a_scales * a.to(dtype=torch.float32)), (b_scales * b.to(dtype=torch.float32))
    ).to(out.dtype)

    out.copy_(c)


# Overwrite autotune_baseline_atol and autotune_baseline_rtol
# if too many configs failed due to baseline check during autotuning
@register_kernel(
    mutates_args=["out"],
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
def block_scaled_mm(
    out: torch.Tensor,  # [M, N]
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    a_scales: torch.Tensor,  # [num_group_m, num_group_k]
    b_scales: torch.Tensor,  # [num_group_k, num_group_n]
) -> None:
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

    assert a_scales.ndim == 2 and b_scales.ndim == 2
    assert a_scales.dtype == b_scales.dtype and a_scales.is_floating_point()

    num_group_m, num_group_k = a_scales.shape
    num_group_n = b_scales.shape[1]
    hl.specialize(num_group_k)
    hl.specialize(num_group_n)

    assert M % num_group_m == 0
    assert K % num_group_k == 0
    assert N % num_group_n == 0

    # a_scales group shape must be [1, 128]
    # b_scales group shape must be [128, 128]
    assert b_scales.shape[0] == num_group_k
    group_m = M // num_group_m
    group_k = K // num_group_k
    group_n = N // num_group_n
    assert group_m == 1 and group_k == 128 and group_n == 128

    hl.specialize(group_k)
    hl.specialize(group_n)

    out_dtype = out.dtype
    assert out_dtype.is_floating_point

    acc_dtype = torch.float32 if a.is_floating_point() else torch.int32
    split_k = hl.register_tunable(
        "split_k", PowerOfTwoFragment(1, helion.next_power_of_2(num_group_k))
    )
    k_block_size = helion.next_power_of_2(helion.cdiv(K, split_k))
    if split_k > 1:
        out.zero_()

    for tile_m, tile_n, outer_k in hl.tile(
        [M, N, K], block_size=[None, None, k_block_size]
    ):
        accumulator = hl.zeros(
            [tile_m, tile_n],
            torch.float32,
        )

        # keep block_size = group_k for K dimension to avoid element-wise scaling
        for tile_k in hl.tile(outer_k.begin, outer_k.end, block_size=group_k):
            a_blk = hl.load(a, [tile_m.index[None, :], tile_k.index[:, None]])
            b_blk = hl.load(b, [tile_k.index[None, :], tile_n.index[:, None]])

            acc_blk = (
                hl.dot(
                    b_blk,
                    a_blk,
                    out_dtype=acc_dtype,
                )
                .t()
                .to(torch.float32)
            )

            gk_idx = tile_k.begin // group_k
            a_scales_blk = a_scales[tile_m, gk_idx][:, None]
            b_scales_blk = b_scales[gk_idx, tile_n.index // group_n][None, :]

            acc_blk = a_scales_blk * acc_blk
            acc_blk = b_scales_blk * acc_blk

            accumulator = accumulator + acc_blk

        out_blk = accumulator.to(out_dtype)

        if split_k == 1:
            out[tile_m, tile_n] = out_blk
        else:
            hl.atomic_add(out, [tile_m, tile_n], out_blk)
