# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up AITER's FlyDSL GEMM for 32x32-block MXFP8 linears on gfx950.

AITER picks a FlyDSL kernel per (padded) token count and JIT-compiles it on
first use. Without a warmup, a cold server compiles several of them inside its
first long prefill step, which can exceed the execute-model RPC timeout.
"""

from __future__ import annotations

import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _flydsl_layers(model: torch.nn.Module):
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        RocmDotScaledMxfp8LinearKernel,
    )

    for module in model.modules():
        # Quark keeps the kernel on the layer's scheme, ModelOpt on the method.
        kernel = getattr(getattr(module, "scheme", None), "ocp_mx_linear", None)
        if kernel is None:
            kernel = getattr(getattr(module, "quant_method", None), "kernel", None)
        if isinstance(kernel, RocmDotScaledMxfp8LinearKernel) and kernel._use_flydsl(
            module
        ):
            yield module


def rocm_flydsl_mxfp8_warmup(model: torch.nn.Module, max_tokens: int) -> None:
    """Run each FlyDSL MXFP8 linear shape once per distinct kernel AITER can
    pick for 1..max_tokens tokens, so serving compiles none of them."""
    from aiter.ops.gemm_op_a8w8 import get_mxscale_bpreshuffle_config

    from vllm._aiter_ops import rocm_aiter_ops

    layers = {tuple(m.weight.shape): m for m in _flydsl_layers(model)}
    if not layers:
        return
    start, num_kernels = time.perf_counter(), 0
    device = next(iter(layers.values())).weight.device
    max_k = max(k for _, k in layers)
    x_buf = torch.zeros(max_tokens * max_k, dtype=torch.float8_e4m3fn, device=device)
    scale_buf = torch.full(
        (max_tokens * max_k // 32,), 127, dtype=torch.uint8, device=device
    ).view(torch.float8_e8m0fnu)
    for (n, k), layer in layers.items():
        x_q = x_buf[: max_tokens * k].view(max_tokens, k)
        x_scale = scale_buf[: max_tokens * k // 32].view(max_tokens, k // 32)
        kernels = set()
        for m in range(1, max_tokens + 1):
            row = get_mxscale_bpreshuffle_config(m, n, k, "32x32")
            if row is not None:
                name = row["kernelName"]
            elif m & (m - 1) == 0 or m == max_tokens:
                # Untuned: AITER's fallback heuristic, sampled at powers of two.
                name = ("untuned", m)
            else:
                continue
            if name in kernels:
                continue
            kernels.add(name)
            rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
                x_q[:m],
                layer.weight,
                x_scale[:m],
                layer.weight_scale.view(torch.float8_e8m0fnu),
                output_dtype=torch.bfloat16,
            )
        num_kernels += len(kernels)
    torch.accelerator.synchronize()
    logger.info(
        "Warmed up %d FlyDSL MXFP8 GEMM kernels for %d shapes in %.1fs.",
        num_kernels,
        len(layers),
        time.perf_counter() - start,
    )
