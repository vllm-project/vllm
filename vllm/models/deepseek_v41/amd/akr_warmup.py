# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill-size JIT warm-up for DeepSeek-V4.1 with amd_kernels routes on.

FlyDSL 0.5 traces a kernel variant before it looks up the on-disk artifact, so
the first launch of every variant in a process costs 0.3-7 s even with a warm
cache. Startup only runs the captured decode sizes and one profile step, so the
variants only prefill-sized steps select (AITER's MXFP8 GEMMs per tuned M bucket,
batched GEMMs, MegaMoE stages, top-k) were traced while serving: one rank stalls
for seconds and, the DP ranks being in lockstep, all of them wait. On a fresh
DEP8 server that was 289 rank-seconds of tracing in the first five minutes.

`prefill_warmup` runs from `kernel_warmup`, after the KV cache exists and before
graph capture:
1. each distinct linear layer (target and draft) runs its `quant_method.apply` at
   one token count per AITER tuned-GEMM bucket above the captured sizes, and so
   does the grouped wo_a; no collectives, so each rank runs it on its own;
2. whole-model dummy steps at a few prefill sizes, the same on every rank.
"""

import time
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.kernel_warmup import _aiter_gemm_m_buckets
from vllm.models.deepseek_v41.amd import akr_hooks

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

MODEL_STEP_SIZES = (1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768)


def gemm_sizes(max_m: int, floor: int) -> list[int]:
    """One token count per AITER tuned-GEMM bucket in (floor, max_m]."""
    return [m for m in _aiter_gemm_m_buckets(max_m) if m > floor]


def model_sizes(max_m: int, floor: int) -> list[int]:
    sizes = {m for m in MODEL_STEP_SIZES if floor < m <= max_m}
    if max_m > floor:
        sizes.add(max_m)
    return sorted(sizes)


def _linear_groups(models: list[torch.nn.Module]) -> list[torch.nn.Module]:
    """One linear layer per distinct quant method and weight geometry."""
    from vllm.model_executor.layers.linear import LinearBase

    seen: dict[tuple, torch.nn.Module] = {}
    for model in models:
        for mod in model.modules():
            w = getattr(mod, "weight", None)
            if (
                not isinstance(mod, LinearBase)
                or getattr(mod, "quant_method", None) is None
                or not isinstance(w, torch.Tensor)
            ):
                continue
            s = getattr(mod, "weight_scale", None)
            key = (
                type(mod.quant_method).__name__,
                int(getattr(mod, "input_size_per_partition", mod.input_size)),
                tuple(w.shape),
                w.dtype,
                tuple(s.shape) if isinstance(s, torch.Tensor) else None,
                bool(getattr(w, "is_shuffled", False)),
            )
            seen.setdefault(key, mod)
    return list(seen.values())


def _warm_linears(layers: list[torch.nn.Module], sizes: list[int], device) -> int:
    calls = 0
    for layer in layers:
        k = int(getattr(layer, "input_size_per_partition", layer.input_size))
        x = torch.randn(sizes[-1], k, dtype=torch.bfloat16, device=device) * 0.1
        for m in sizes:
            try:
                layer.quant_method.apply(layer, x[:m], bias=None)
            except Exception as e:
                logger.debug("prefill warm-up skips %s: %s", type(layer).__name__, e)
                break
            calls += 1
    return calls


def _warm_wo_a(models: list[torch.nn.Module], sizes: list[int], device) -> int:
    """The stock grouped MXFP8 wo_a is not a `LinearBase.apply`."""
    from vllm.platforms import current_platform
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_mxfp8_wo_a_bmm

    seen: set[tuple] = set()
    calls = 0
    for model in models:
        for mod in model.modules():
            wo_a = getattr(mod, "wo_a", None)
            groups = getattr(mod, "n_local_groups", None)
            rank = getattr(mod, "o_lora_rank", None)
            w = getattr(wo_a, "weight", None)
            if not isinstance(w, torch.Tensor) or not groups or not rank:
                continue
            key = (tuple(w.shape), w.dtype, groups, rank)
            if w.element_size() != 1 or key in seen:
                continue
            seen.add(key)
            width = groups * w.shape[1]
            a = torch.zeros(
                sizes[-1], width, dtype=current_platform.fp8_dtype(), device=device
            )
            s = torch.full(
                (sizes[-1], width // 32), 127, dtype=torch.uint8, device=device
            )
            for m in sizes:
                rocm_mxfp8_wo_a_bmm(a[:m], s[:m], wo_a, groups, rank)
                calls += 1
    return calls


@torch.inference_mode()
def prefill_warmup(worker: "Worker") -> None:
    if (
        not akr_hooks.loaded
        or not worker.vllm_config.kernel_config.amd_kernels.prefill_warmup
    ):
        return
    runner = worker.model_runner
    max_m = worker.scheduler_config.max_num_batched_tokens
    floor = worker.vllm_config.compilation_config.max_cudagraph_capture_size or 0
    start = time.perf_counter()
    models = [worker.get_model()]
    draft = worker.get_draft_model()
    if draft is not None:
        models.append(draft)
    layers = _linear_groups(models)
    sizes = gemm_sizes(max_m, floor)
    calls = 0
    if sizes:
        calls = _warm_linears(layers, sizes, worker.device)
        calls += _warm_wo_a(models, sizes, worker.device)
    torch.accelerator.synchronize()
    gemm_end = time.perf_counter()
    steps = model_sizes(max_m, floor)
    for n in steps:
        runner._dummy_run(
            n, skip_eplb=True, randomize_inputs=worker.randomize_dummy_inputs
        )
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    logger.info(
        "amd_kernels prefill warm-up: %d GEMM calls (%d linear groups, %d M "
        "buckets) in %.1fs, model steps %s in %.1fs",
        calls,
        len(layers),
        len(sizes),
        gemm_end - start,
        steps,
        time.perf_counter() - gemm_end,
    )
