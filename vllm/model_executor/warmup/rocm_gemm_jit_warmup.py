# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile before serving the Triton variants that ROCm's unquantized GEMM dispatch
and the bf16x3 MoE router would otherwise JIT-compile inside the first requests.

Triton specializes integer arguments on ``== 1`` and on divisibility by 16, and the
startup forwards (profile run, CUDA-graph capture) use token counts that are mostly
multiples of 16. The first real prefill or mixed step with another count then compiles
another variant in the middle of the step. On MiniMax-M3 (gfx950) these are the bf16x3
router reduction at >= 2048 tokens and AITER's tuned Triton GEMM of the dense MLP, each
about a second, which shows up as a TTFT spike on the first requests after a server
start. This calls the same dispatchers the layers use, at token counts on both sides
of every specialization boundary, so whatever Triton kernel they pick is compiled here.
"""

import time
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

# 1-32 tokens (skinny kernels, tuned rows padded to 16 / 32) and both sides of the
# powers of two up to 4096 (the bf16x3 router starts at 2048 tokens).
_TOKEN_COUNTS = (
    tuple(range(1, 33))
    + tuple(m for p in (64, 128, 256, 512, 1024, 2048, 4096) for m in (p - 1, p, p + 1))
    + (2064,)
)


def _enabled() -> bool:
    import vllm.envs as envs

    return envs.VLLM_ROCM_GEMM_JIT_WARMUP


def rocm_gemm_jit_warmup(worker: "Worker") -> None:
    if not (current_platform.is_rocm() and _enabled()):
        return

    import vllm.envs as envs
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    if envs.VLLM_BATCH_INVARIANT:
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    counts = sorted(m for m in _TOKEN_COUNTS if m <= max_tokens)
    seen: set[tuple[Any, ...]] = set()
    calls = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for module in worker.get_model().modules():
            if isinstance(module, GateLinear):
                weight_split = getattr(module, "_bf16x3_weight", None)
                if not getattr(module, "allow_rocm_bf16x3_router_gemm", False) or (
                    weight_split is None
                ):
                    continue
                weight = module.weight
                key: tuple[Any, ...] = ("router", tuple(weight.shape), weight.dtype)
                if key in seen:
                    continue
                seen.add(key)
                for m in counts:
                    x = torch.zeros(
                        m, weight.shape[1], dtype=torch.bfloat16, device=weight.device
                    )
                    torch.ops.vllm.rocm_bf16x3_router_gemm_dispatch(
                        x, weight, weight_split
                    )
                    calls += 1
                continue
            if not isinstance(module, LinearBase) or not isinstance(
                getattr(module, "quant_method", None), UnquantizedLinearMethod
            ):
                continue
            weight = getattr(module, "weight", None)
            if (
                not isinstance(weight, torch.Tensor)
                or weight.dim() != 2
                or weight.dtype not in (torch.bfloat16, torch.float16)
                or not weight.is_cuda
            ):
                continue
            bias = getattr(module, "bias", None)
            # Row-parallel layers add the bias on rank 0 only: warm both variants.
            for b in (None, bias) if isinstance(bias, torch.Tensor) else (None,):
                key = (
                    "linear",
                    tuple(weight.shape),
                    tuple(weight.stride()),
                    weight.dtype,
                    b is not None,
                )
                if key in seen:
                    continue
                seen.add(key)
                for m in counts:
                    x = torch.zeros(
                        m, weight.shape[1], dtype=weight.dtype, device=weight.device
                    )
                    torch.ops.vllm.rocm_unquantized_gemm(x, weight, b)
                    calls += 1
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    logger.info(
        "ROCm GEMM JIT warmup: %d calls over %d weight shapes, token counts %d-%d, "
        "in %.2fs.",
        calls,
        len(seen),
        counts[0],
        counts[-1],
        time.perf_counter() - start,
    )
