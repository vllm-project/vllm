# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deployment qualification for the Qwen3 RDNA dense specialized decode."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.platforms import current_platform


@dataclass(frozen=True, slots=True)
class DecodeConfig:
    lib_name: str = "qwen3_1_7b"
    rms_norm_eps: float = 1e-6
    qk_norm: bool = True


def specialized_decode_config(vllm_config: VllmConfig) -> DecodeConfig | None:
    if not vllm_config.kernel_config.monokernel:
        return None

    from vllm.platforms.rocm import on_rdna

    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    cache = vllm_config.cache_config
    config = model.hf_config.get_text_config()
    reasons: list[str] = []

    if not current_platform.is_rocm():
        reasons.append("requires ROCm")
    elif not on_rdna():
        reasons.append("requires RDNA")

    if parallel.tensor_parallel_size != 1:
        reasons.append("requires tensor parallel size 1")
    if parallel.pipeline_parallel_size != 1:
        reasons.append("requires pipeline parallel size 1")
    if parallel.data_parallel_size != 1:
        reasons.append("requires data parallel size 1")
    if parallel.decode_context_parallel_size != 1:
        reasons.append("requires decode context parallel size 1")
    if parallel.prefill_context_parallel_size != 1:
        reasons.append("requires prefill context parallel size 1")
    if parallel.use_ubatching or parallel.enable_dbo:
        reasons.append("does not support DBO or ubatching")
    if parallel.enable_expert_parallel or parallel.enable_eplb:
        reasons.append("does not support expert parallelism or EPLB")

    if model.dtype != torch.bfloat16:
        reasons.append("requires BF16 weights and activations")
    if vllm_config.quant_config is not None:
        reasons.append("does not support quantized weights")
    if cache.cache_dtype not in ("auto", "bfloat16"):
        reasons.append("requires BF16 KV cache")
    if vllm_config.speculative_config is not None:
        reasons.append("does not support speculative decoding")
    if vllm_config.lora_config is not None:
        reasons.append("does not support LoRA")
    if model.enable_sleep_mode:
        reasons.append("does not support sleep mode")

    if getattr(config, "qk_norm", True) is not True:
        reasons.append("requires qk_norm=True")
    if getattr(config, "hidden_act", None) != "silu":
        reasons.append("requires SiLU activation")

    if reasons:
        raise ValueError(
            "Qwen3 RDNA dense specialized decode unsupported: " + "; ".join(reasons)
        )
    return DecodeConfig()
