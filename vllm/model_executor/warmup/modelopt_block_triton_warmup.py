# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm the ModelOpt block Triton kernels used by a loaded model.

The dense dispatch picks a tile from the token count inside the custom op, so a
prefill chunk of any size can reach a config that the dummy runs never touched.
Compile every distinct config once, before CUDA graph capture, so inference does
not JIT.
"""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _max_tokens(worker: "Worker") -> int:
    """The largest token count one forward pass can see."""
    scheduler = worker.scheduler_config
    compilation = worker.vllm_config.compilation_config
    sizes = [
        1,
        scheduler.max_num_batched_tokens,
        scheduler.max_num_seqs,
        *(compilation.cudagraph_capture_sizes or []),
        *(s for s in (compilation.compile_sizes or []) if isinstance(s, int)),
    ]
    return max(int(size) for size in sizes)


def _triton_layers(model: torch.nn.Module):
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if method is not None and getattr(method, "use_triton", False):
            yield module, method


@torch.inference_mode()
def modelopt_block_triton_warmup(worker: "Worker") -> None:
    """Compile every ModelOpt block config reachable at serving token counts."""
    if not current_platform.is_cuda():
        return

    model = worker.get_model()
    layers = list(_triton_layers(model))
    if not layers:
        return

    from vllm.model_executor.layers.quantization import (
        modelopt_block_quant_triton as kernels,
    )

    max_tokens = _max_tokens(worker)
    props = torch.cuda.get_device_properties(worker.device)
    dtype = worker.vllm_config.model_config.dtype
    num_sms = props.multi_processor_count
    smem = props.shared_memory_per_block_optin

    warmed = 0
    seen_linear: set[tuple] = set()
    seen_dequant: set[tuple] = set()
    seen_moe: set[tuple] = set()

    for module, method in layers:
        codec = method.codec
        if hasattr(module, "w13_weight"):
            _warm_moe(
                module,
                method,
                kernels,
                max_tokens,
                seen_moe,
                dtype,
            )
            warmed += 1
            continue

        device = module.weight.device
        if getattr(method, "is_embedding", False):
            for id_dtype in (torch.int64, torch.int32):
                ids = torch.zeros(1, dtype=id_dtype, device=device)
                torch.ops.vllm.modelopt_block_embedding(
                    ids, module.weight, module.weight_scale, dtype
                )
            warmed += 1
            continue

        qweight = module.weight
        n = qweight.shape[0]
        k = module.weight_scale.shape[1] * method.block_size
        fused_configs: dict[tuple, int] = {}
        min_m = kernels.dequant_min_m(codec, device.index)
        fused_max = max_tokens
        if n * k <= kernels.DEQUANT_MAX_NUMEL:
            fused_max = min(fused_max, min_m - 1)
            key: tuple = (codec, n, k, dtype)
            if max_tokens >= min_m and key not in seen_dequant:
                seen_dequant.add(key)
                x = torch.zeros((min_m, k), dtype=dtype, device=device)
                method.apply(module, x)
                warmed += 1
        # Above 64 rows only the count of M tiles can change split-K.
        bm = kernels._linear_config(65, n, k, codec, num_sms, smem, props.major)["BM"]
        counts = set(range(1, min(fused_max, 65) + 1))
        counts.update(
            m + delta
            for m in range(bm + 1, fused_max + 1, bm)
            for delta in (0, 1)
            if m + delta <= fused_max
        )
        for m in sorted(counts):
            cfg = kernels._linear_config(m, n, k, codec, num_sms, smem, props.major)
            # The split-K sum's size argument (m * n) is specialized on % 16.
            key = (codec, n, k, dtype, (m * n) % 16 == 0, tuple(sorted(cfg.items())))
            if key not in seen_linear:
                seen_linear.add(key)
                fused_configs[key] = m
        for m in fused_configs.values():
            x = torch.zeros((m, k), dtype=dtype, device=device)
            torch.ops.vllm.modelopt_block_linear(
                x,
                qweight,
                module.weight_scale,
                getattr(module, "iq2_lut", None),
                codec,
            )
            warmed += 1

    if warmed:
        torch.accelerator.synchronize()
        logger.info("Warmed %d ModelOpt block Triton configs.", warmed)


@torch.inference_mode()
def _warm_moe(
    module: torch.nn.Module,
    method,
    kernels,
    max_tokens: int,
    seen: set[tuple],
    dtype: torch.dtype,
) -> None:
    e, k = module.w2_weight.shape[:2]
    top_k = module.top_k
    # Configs change at 8 rows (GEMV), at 1 and 64 rows per expert
    # (`moe_config`) and where `_prepare_expert_assignment` stops skipping the
    # token sort (M * top_k * 4 <= E); Triton also specializes on M == 1.
    # Warm through `method.apply`, the same pipeline as inference.
    counts = {
        1,
        2,
        8,
        9,
        e // (4 * top_k),
        e // (4 * top_k) + 1,
        (e + top_k - 1) // top_k,
        (64 * e + top_k - 1) // top_k,
    }
    for m in sorted(n for n in counts if 0 < n <= max_tokens):
        gemv = kernels.moe_gemv_config(m, e, top_k, method.codec)
        cfg = gemv or kernels.moe_config(m, e, top_k)
        key = (
            method.codec,
            module.w13_weight.shape,
            module.w2_weight.shape,
            dtype,
            module.activation,
            top_k,
            module.apply_router_weight_on_input,
            gemv is not None,
            m * top_k * 4 <= e,
            tuple(sorted(cfg.items())),
        )
        if key in seen:
            continue
        seen.add(key)
        x = torch.zeros((m, k), dtype=dtype, device=module.w2_weight.device)
        ids = torch.arange(top_k, dtype=torch.int32, device=x.device)
        ids = ids.expand(m, -1).contiguous()
        weights = torch.full((m, top_k), 1 / top_k, device=x.device)
        method.apply(module, x, weights, ids, None, None)
