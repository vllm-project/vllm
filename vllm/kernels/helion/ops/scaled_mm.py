# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from itertools import product
from typing import Any

import helion
import helion.language as hl
import regex as re
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


def generate_inputs() -> dict[str, tuple[Any, ...]]:
    # TODO(xiaohongchen1991): it is difficult for kernel author to cover
    # all input property combination. Currently, dtypes are fixed. We need
    # optimization to bucket/skip some combinations
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
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

        # Qwen3.5-35B-A3B
        (2048, 1024),
        (512, 2048),
        (2048, 9216),
        (4096, 2048),

        # Meta-Llama-3.3-70B
        (8192, 10240),
        (8192, 8192),
        (8192, 57344),
        (28672, 8192),
    ]

    in_dtype: torch.dtype = current_platform.fp8_dtype()
    scale_dtype: torch.dtype = torch.float32
    out_dtype: torch.dtype = torch.bfloat16
    inputs = {}
    for num_tokens, (hidden_size, feature_size) in product(
        num_tokens_list, b_shape_list
    ):
        scale = 1.0 / math.sqrt(hidden_size)
        a = (
            scale
            * (
                0.5
                + torch.rand(
                    num_tokens, hidden_size, dtype=torch.float32, device="cuda"
                )
            )
        ).to(in_dtype)
        b = (
            scale
            * (
                0.5
                + torch.rand(
                    feature_size, hidden_size, dtype=torch.float32, device="cuda"
                )
            )
        ).to(in_dtype)
        b = b.t()
        scale_a = 0.5 + torch.rand((num_tokens, 1), dtype=scale_dtype, device="cuda")
        scale_b = 0.5 + torch.rand((feature_size, 1), dtype=scale_dtype, device="cuda")
        bias = 0.5 * (torch.rand(feature_size, dtype=out_dtype, device="cuda") - 0.5)

        config_key = (
            f"hidden_size_{hidden_size}_"
            f"feature_size_{feature_size}_num_tokens_{num_tokens}"
        )
        inputs[config_key] = (a, b, scale_a, scale_b, out_dtype, bias)

    return inputs


def pick_config(args: tuple[Any, ...], config_keys: list[str]) -> str | None:
    """Pick the best pre-tuned config for the given input shape.
    Selection strategy:
      1. Find the closest hidden_size among available configs
         (exact match preferred).
      2. Find the closest feature_size among available configs
         (exact match preferred).
      3. Among the num_tokens values tuned for that hidden_size and feature_size, pick
         the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.
    Config keys must be "default" or follow the format
    "hidden_size_{int}_feature_size_{int}_num_tokens_{int}".
    """

    if not config_keys:
        return None

    a, b, *_ = args
    num_tokens, hidden_size = a.shape
    feature_size = b.shape[1]

    configs: dict[int, dict[int, list[int]]] = {}
    for key in config_keys:
        if key == "default":
            continue
        match = re.fullmatch(
            r"hidden_size_(\d+)_feature_size_(\d+)_num_tokens_(\d+)", key
        )
        if not match:
            raise ValueError(
                f"Malformed config key '{key}', "
                f"expected format 'hidden_size_{{int}}_"
                f"feature_size_{{int}}_num_tokens_{{int}}'"
            )
        hidden_size_str, feature_size_str, num_tokens_str = match.groups()
        configs.setdefault(int(hidden_size_str), {}).setdefault(
            int(feature_size_str), []
        ).append(int(num_tokens_str))

    if not configs:
        return "default" if "default" in config_keys else None

    best_hidden_size = min(configs, key=lambda s: abs(s - hidden_size))
    best_feature_size = min(
        configs[best_hidden_size], key=lambda s: abs(s - feature_size)
    )
    available_num_tokens = sorted(configs[best_hidden_size][best_feature_size])
    best_num_tokens = next(
        (n for n in available_num_tokens if n >= num_tokens), available_num_tokens[-1]
    )

    return (
        f"hidden_size_{best_hidden_size}_feature_size_"
        f"{best_feature_size}_num_tokens_{best_num_tokens}"
    )


def fake_impl(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    scale_b: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,  # [N]
) -> torch.Tensor:
    M = a.shape[0]
    N = b.shape[1]
    c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    return c


@register_kernel(
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_baseline_atol=1.0,
        autotune_baseline_rtol=5e-1,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ),
)  # type: ignore[misc]
def scaled_mm(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    scale_b: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,  # [N]
) -> torch.Tensor:
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
    assert out_dtype.is_floating_point

    if bias is not None:
        assert bias.numel() == N and bias.dtype == out_dtype

    c = torch.empty((M, N), dtype=out_dtype, device=a.device)
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
        scale_a_mask = (tile_m.index < scale_a.shape[0])[:, None]
        scale_a_blk = torch.where(scale_a_mask, scale_a[tile_m, :], scale_a[0, 0])
        acc = scale_a_blk * acc

        scale_b_mask = (tile_n.index < scale_b.shape[0])[:, None]
        scale_b_blk = torch.where(scale_b_mask, scale_b[tile_n, :], scale_b[0, 0])
        acc = scale_b_blk.T * acc

        c_blk = acc.to(out_dtype)

        if bias is not None:
            c_blk += bias[tile_n]

        hl.store(c, [tile_m, tile_n], c_blk, extra_mask=(tile_m.index < M)[:, None])

    return c


def baseline(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    scale_b: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,  # [N]
) -> torch.Tensor:
    out = torch.mm(a.to(torch.float32), b.to(torch.float32))
    out = scale_a * out
    out = scale_b.T * out
    out = out.to(out_dtype)
    if bias is not None:
        out = out + bias

    return out
