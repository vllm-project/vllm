# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.internvl_shuffle_layer_norm import (
    internvl_shuffle_layer_norm,
)
from vllm.triton_utils import triton

_SHAPES = (
    (1, 32, 32, 1024),
    (4, 32, 32, 1024),
    (16, 32, 32, 1024),
    (32, 32, 32, 1024),
    (4, 32, 24, 1024),
)


def baseline(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    batch, height, width, channels = x.shape
    shuffled = x.view(batch, height, width // 2, channels * 2)
    shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
    shuffled = shuffled.view(batch, width // 2, height // 2, channels * 4)
    shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
    return F.layer_norm(
        shuffled,
        (shuffled.shape[-1],),
        weight,
        bias,
        eps,
    )


def measure(
    fn: Callable[[], torch.Tensor],
    warmup_ms: int,
    repeat_ms: int,
) -> float:
    return 1000 * triton.testing.do_bench(
        fn,
        warmup=warmup_ms,
        rep=repeat_ms,
        return_mode="median",
    )


@torch.inference_mode()
def main(dtype: torch.dtype, warmup_ms: int, repeat_ms: int) -> None:
    torch.manual_seed(0)
    print(
        "batch,height,width,channels,dtype,baseline_us,fused_us,"
        "speedup,max_abs,mean_abs"
    )
    for shape in _SHAPES:
        batch, height, width, channels = shape
        with_class_token = torch.randn(
            batch,
            height * width + 1,
            channels,
            device="cuda",
            dtype=dtype,
        )
        x = with_class_token[:, 1:].view(shape)
        weight = torch.randn(4 * channels, device="cuda", dtype=dtype)
        bias = torch.randn_like(weight)
        args = (x, weight, bias, 1e-5)
        functions = {
            "baseline": partial(baseline, *args),
            "fused": partial(internvl_shuffle_layer_norm, *args),
        }

        expected = functions["baseline"]()
        actual = functions["fused"]()
        error = (actual.float() - expected.float()).abs()
        baseline_us = measure(functions["baseline"], warmup_ms, repeat_ms)
        fused_us = measure(functions["fused"], warmup_ms, repeat_ms)
        print(
            f"{batch},{height},{width},{channels},{dtype},"
            f"{baseline_us:.3f},{fused_us:.3f},"
            f"{baseline_us / fused_us:.4f},{error.max().item():.8f},"
            f"{error.mean().item():.8f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--repeat-ms", type=int, default=300)
    args = parser.parse_args()
    main(getattr(torch, args.dtype), args.warmup_ms, args.repeat_ms)
