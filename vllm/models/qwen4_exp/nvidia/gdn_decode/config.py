# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent opt-in for the measured SM120 GDN output-tail fusion."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import Qwen4ExpTextConfig

    from vllm.config import ParallelConfig, VllmConfig


def gdn_decode_enabled(additional_config: object) -> bool:
    if not isinstance(additional_config, dict):
        return False
    enabled = additional_config.get("gdn_norm_quant_decode", False)
    if type(enabled) is not bool:
        raise ValueError("gdn_norm_quant_decode must be a boolean")
    return enabled


def validate_gdn_decode(vllm_config: "VllmConfig") -> None:
    from vllm import envs
    from vllm.config import CompilationMode, CUDAGraphMode
    from vllm.platforms import current_platform

    config = vllm_config.compilation_config
    if (
        config.mode != CompilationMode.NONE
        or config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
    ):
        raise ValueError(
            "gdn_norm_quant_decode requires CompilationMode.NONE and "
            "FULL_DECODE_ONLY CUDA graphs"
        )
    capability = (
        current_platform.get_device_capability() if current_platform.is_cuda() else None
    )
    offload = vllm_config.offload_config
    validate_decode_deployment(
        vllm_config.model_config.hf_text_config,
        vllm_config.parallel_config,
        dtype=vllm_config.model_config.dtype,
        quantization=(
            vllm_config.quant_config.get_name()
            if vllm_config.quant_config is not None
            else None
        ),
        device_capability=(
            (capability.major, capability.minor) if capability is not None else None
        ),
        has_lora=vllm_config.lora_config is not None,
        has_speculative=vllm_config.speculative_config is not None,
        has_weight_offload=(
            offload.uva.cpu_offload_gb > 0 or offload.prefetch.offload_group_size > 0
        ),
        enable_sleep_mode=vllm_config.model_config.enable_sleep_mode,
        batch_invariant=envs.VLLM_BATCH_INVARIANT,
    )


def validate_decode_deployment(
    text_config: "Qwen4ExpTextConfig",
    parallel_config: "ParallelConfig",
    *,
    dtype: "torch.dtype",
    quantization: str | None,
    device_capability: tuple[int, int] | None,
    has_lora: bool = False,
    has_speculative: bool = False,
    has_weight_offload: bool = False,
    enable_sleep_mode: bool = False,
    enable_hc_sp: bool = False,
    batch_invariant: bool = False,
) -> None:
    """Reject unsupported static configurations before allocating kernel scratch."""
    import torch

    errors = []
    expected = {
        "hidden_size": 2560,
        "num_experts": 512,
        "moe_intermediate_size": 640,
        "num_experts_per_tok": 10,
        "shared_expert_intermediate_size": 640,
        "hidden_act": "silu",
        "norm_topk_prob": True,
    }
    for name, value in expected.items():
        if getattr(text_config, name, None) != value:
            errors.append(f"{name}={value}")
    if dtype != torch.bfloat16:
        errors.append("BF16 model dtype")
    if quantization != "modelopt_mixed":
        errors.append("ModelOpt mixed quantization with routed NVFP4 experts")
    if device_capability != (12, 0):
        errors.append("an SM120 CUDA device")
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if getattr(parallel_config, name) != 1:
            errors.append(f"{name}=1")
    if parallel_config.ubatch_size > 1:
        errors.append("no microbatching")
    for name in (
        "enable_expert_parallel",
        "use_sequence_parallel_moe",
        "enable_eplb",
        "enable_dbo",
        "enable_hc_sp",
    ):
        if getattr(parallel_config, name):
            errors.append(f"{name}=False")
    if parallel_config.eplb_config.num_redundant_experts:
        errors.append("no redundant experts")
    for enabled, requirement in (
        (has_lora, "no LoRA"),
        (has_speculative, "no speculative decoding"),
        (has_weight_offload, "no generic model weight offload"),
        (enable_sleep_mode, "no sleep mode (decode buffers are not restored)"),
        (enable_hc_sp, "no HC sequence parallelism"),
        (batch_invariant, "VLLM_BATCH_INVARIANT disabled"),
    ):
        if enabled:
            errors.append(requirement)
    if errors:
        raise ValueError(
            "Qwen4Exp gdn_norm_quant_decode requires: " + ", ".join(errors)
        )
