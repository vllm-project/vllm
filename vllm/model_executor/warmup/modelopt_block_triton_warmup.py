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


def _warmup_token_counts(worker: "Worker") -> int:
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

    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.quantization import (
        modelopt_block_quant_triton as kernels,
    )

    max_tokens = _warmup_token_counts(worker)
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
                moe_align_block_size,
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
        for m in range(1, max_tokens + 1):
            if m >= min_m and n * k <= kernels.DEQUANT_MAX_NUMEL:
                key: tuple = (codec, n, k)
                if key not in seen_dequant:
                    seen_dequant.add(key)
                    x = torch.zeros((min_m, k), dtype=dtype, device=device)
                    torch.ops.vllm.modelopt_block_linear(
                        x,
                        qweight,
                        module.weight_scale,
                        getattr(module, "iq2_lut", None),
                        codec,
                        getattr(module, "weight_bf16", None),
                    )
                    warmed += 1
                continue
            cfg = kernels._linear_config(m, n, k, codec, num_sms, smem, props.major)
            key = (codec, n, k, tuple(sorted(cfg.items())))
            if key not in seen_linear:
                seen_linear.add(key)
                fused_configs[key] = m
        for key, m in fused_configs.items():
            x = torch.zeros((m, k), dtype=dtype, device=device)
            torch.ops.vllm.modelopt_block_linear(
                x,
                qweight,
                module.weight_scale,
                getattr(module, "iq2_lut", None),
                codec,
                getattr(module, "weight_bf16", None),
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
    moe_align_block_size,
    max_tokens: int,
    seen: set[tuple],
    dtype: torch.dtype,
) -> None:
    codec = method.codec
    device = module.w13_weight.device
    num_experts = module.w13_weight.shape[0]
    top_k = module.top_k
    warm_k = module.w2_weight.shape[1]
    n1 = module.w13_weight.shape[1]
    n2 = module.w2_weight.shape[1]
    k2 = module.w2_weight_scale.shape[2] * method.block_size
    mul_routed_weight = not module.apply_router_weight_on_input
    lut = kernels.iq2_lut(device) if codec == "iq2_xxs" else None

    configs: dict[tuple, tuple[dict[str, int], bool]] = {}
    for m in range(1, max_tokens + 1):
        gemv = kernels.moe_gemv_config(m, num_experts, top_k, codec)
        cfg = gemv or kernels.moe_config(m, num_experts, top_k)
        key = (
            codec,
            n1,
            warm_k,
            n2,
            k2,
            top_k,
            mul_routed_weight,
            gemv is not None,
            module.expert_map is None,
            module.activation,
            tuple(sorted(cfg.items())),
        )
        if key in seen:
            continue
        seen.add(key)
        configs[key] = (cfg, gemv is not None)

    for key, (cfg, gemv) in configs.items():
        launch = (
            kernels.invoke_block_moe_gemv_kernel
            if gemv
            else kernels.invoke_block_moe_kernel
        )
        # One aligned block per routed row on the GEMV path; the token count there
        # only sizes the dummy routing, the kernel takes no tile in M.
        bm = 1 if gemv else cfg["BLOCK_SIZE_M"]
        rows = 2 if gemv else bm
        topk_ids = torch.randint(
            0, num_experts, (rows, top_k), dtype=torch.int32, device=device
        )
        topk_weights = torch.rand((rows, top_k), dtype=torch.float32, device=device)
        routing = (
            (None, topk_ids, None)
            if gemv and module.expert_map is None
            else moe_align_block_size(topk_ids, bm, num_experts, module.expert_map)
        )
        a1 = torch.zeros((rows, warm_k), dtype=dtype, device=device)
        c1 = torch.empty((rows, top_k, n1), dtype=dtype, device=device)
        launch(
            a1,
            module.w13_weight,
            module.w13_weight_scale,
            lut,
            c1,
            None,
            *routing,
            False,
            top_k,
            cfg,
            codec,
            # MoEActivation is a plain Enum: compare its value, not the member.
            **(
                {"apply_relu2": module.activation.value == "relu2_no_mul"}
                if gemv
                else {}
            ),
        )
        a2 = torch.empty((rows * top_k, k2), dtype=dtype, device=device)
        c2 = torch.empty((rows, top_k, n2), dtype=dtype, device=device)
        launch(
            a2,
            module.w2_weight,
            module.w2_weight_scale,
            lut,
            c2,
            topk_weights,
            *routing,
            mul_routed_weight,
            1,
            cfg,
            codec,
        )
        del topk_ids, topk_weights, routing, a1, c1, a2, c2
