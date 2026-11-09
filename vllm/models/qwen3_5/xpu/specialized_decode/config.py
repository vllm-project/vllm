# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deployment qualification for the opt-in XPU MoE executor."""

import torch
import torch.distributed as dist

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_tp_group
from vllm.platforms import current_platform

from . import is_requested


def agree(reasons: list[str]) -> None:
    """Collect initialization failures before any TP rank starts execution."""
    group = get_tp_group()
    failures: list[list[str] | None] = [None] * group.world_size
    dist.all_gather_object(failures, reasons, group=group.cpu_group)
    missing = [
        f"TP rank {rank}: {reason}"
        for rank, errors in enumerate(failures)
        for reason in errors or ()
    ]
    if missing:
        raise ValueError(
            "Qwen3.5 decode_backend=latency_specialized is unsupported:\n"
            + "\n".join(missing)
            + "\nUse decode_backend=auto for the default XPU implementation."
        )


def validate_deployment(config: VllmConfig) -> None:
    """Qualify the measured deployment before constructing model weights."""
    if not is_requested(config):
        return
    model, parallel, cache = (
        config.model_config,
        config.parallel_config,
        config.cache_config,
    )
    text = model.hf_text_config
    offload = config.offload_config
    checks = (
        (not current_platform.is_xpu(), "requires XPU"),
        (
            tuple(
                getattr(text, name, None)
                for name in (
                    "hidden_size",
                    "moe_intermediate_size",
                    "shared_expert_intermediate_size",
                    "num_experts",
                    "num_experts_per_tok",
                )
            )
            != (2048, 512, 512, 256, 8),
            (
                "requires validated MoE dimensions "
                "(H=2048, I=512, shared_I=512, E=256, top_k=8)"
            ),
        ),
        (
            getattr(text, "num_hidden_layers", None) != 40,
            "requires the validated 40-layer MoE architecture",
        ),
        (parallel.tensor_parallel_size != 2, "requires TP2"),
        (
            parallel.distributed_executor_backend != "mp",
            "requires the validated single-host mp executor",
        ),
        (
            parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.enable_expert_parallel
            or parallel.enable_eplb
            or parallel.use_sequence_parallel_moe,
            "PP, DP, context/sequence/expert parallelism and EPLB are unqualified",
        ),
        (
            model.dtype != torch.float16,
            "the measured deployment requires FP16 activations",
        ),
        (
            model.quantization != "fp8_per_tensor",
            "requires fp8_per_tensor quantization",
        ),
        (
            model.multimodal_config is not None
            and not model.multimodal_config.language_model_only,
            "multimodal inputs are unqualified",
        ),
        (config.lora_config is not None, "LoRA is unqualified"),
        (config.speculative_config is not None, "speculative decoding is unqualified"),
        (cache.enable_prefix_caching, "prefix caching is unqualified"),
        (
            cache.cache_dtype != "auto"
            or cache.mamba_cache_dtype != "auto"
            or cache.mamba_ssm_cache_dtype not in ("auto", "float32"),
            "requires auto KV/conv dtype and FP32 recurrent state",
        ),
        (model.enable_sleep_mode, "sleep/wake is unqualified"),
        (model.enable_prompt_embeds, "user-provided prompt embeddings are unqualified"),
        (
            cache.use_replayssm or cache.kv_sharing_fast_prefill,
            "alternate KV/SSM execution paths are unqualified",
        ),
        (
            bool(cache.enable_extensible_kv_cache) or bool(cache.kv_offloading_size),
            "KV rebinding/offloading is unqualified",
        ),
        (config.kv_transfer_config is not None, "KV transfer/rebinding is unqualified"),
        (
            config.weight_transfer_config is not None,
            "external weight transfer is unqualified",
        ),
        (
            offload.uva.cpu_offload_gb > 0 or offload.prefetch.offload_group_size > 0,
            "weight offloading is unqualified",
        ),
        (parallel.enable_dbo or parallel.ubatch_size > 1, "DBO is unqualified"),
        (config.kernel_config.linear_backend != "auto", "requires linear_backend=auto"),
        (config.compilation_config.mode != 0, "requires eager compilation mode=0"),
        (
            config.compilation_config.cudagraph_mode
            not in (
                CUDAGraphMode.NONE,
                CUDAGraphMode.FULL_DECODE_ONLY,
            ),
            "requires graph mode NONE or FULL_DECODE_ONLY",
        ),
        (model.max_model_len > 40960, "context lengths above 40960 are unqualified"),
    )
    reasons = [reason for failed, reason in checks if failed]
    if current_platform.is_xpu():
        device = torch.device("xpu", torch.xpu.current_device())
        props = torch.xpu.get_device_properties(device)
        if props.device_id != 0xE223:
            reasons.append("requires the validated Arc Pro B70 SKU (device_id=0xE223)")
        if not (props.has_fp16 and props.has_subgroup_2d_block_io):
            reasons.append("requires FP16 and subgroup block-2D capabilities")
        try:
            from vllm_xpu_kernels.moe_shared_fused_interface import supports
        except ImportError:
            reasons.append("the optional shared-expert fused provider is unavailable")
        else:
            if not supports(
                torch.float16, torch.float8_e4m3fn, 8, 2048, 256, device=device
            ):
                reasons.append(
                    "the installed fused provider does not support this device"
                )
    agree(reasons)
