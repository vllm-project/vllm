# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash specialized decode.

Supported Blackwell and Hopper deployments use specialized decode by default.
`decode_backend=stock` disables specialized decode.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DecodeConfig:
    max_context: int
    # Tokens per request in a verify step (1 + draft tokens; 1 without speculation).
    q: int
    capture_sizes: tuple[int, ...]
    # Overrides of the per-token-count dispatch (see ``policy.DEFAULT``).
    policy: tuple[tuple[str, str], ...] = ()


def specialized_decode_config(vllm_config) -> DecodeConfig | None:
    """The deployment's specialized decode settings, None for the stock decode.
    Auto falls back for unsupported deployments; latency_specialized raises."""
    extra = vllm_config.additional_config
    extra = extra if isinstance(extra, dict) else {}
    backend = extra.get("decode_backend", "auto")
    if backend == "stock":
        return None
    if backend not in ("auto", "latency_specialized"):
        raise ValueError(
            "Supported decode_backend values are auto/stock/latency_specialized"
        )
    from ..policy import platform_policy, unsupported_platform

    policy = dict(extra.get("policy", {}))
    reasons = _unsupported_deployment(vllm_config) + unsupported_platform(
        vllm_config, policy
    )
    if reasons:
        if backend == "auto":
            return None
        raise ValueError(
            "GLM-5.3-Flash specialized decode does not support " + "; ".join(reasons)
        )
    policy = platform_policy(
        vllm_config, policy, strict=backend == "latency_specialized"
    )
    spec = vllm_config.speculative_config
    return DecodeConfig(
        max_context=vllm_config.model_config.max_model_len,
        q=1 + spec.num_speculative_tokens if spec is not None else 1,
        capture_sizes=tuple(
            vllm_config.compilation_config.cudagraph_capture_sizes or ()
        ),
        policy=tuple(policy.items()),
    )


# The model geometry the kernels are specialized for: hidden size, KDA heads and
# head dim, conv width, MLA heads / latent / NoPE dims, indexer, mHC streams, MoE.
GEOMETRY = {
    "hidden_size": 4096,
    "linear_num_heads": 64,
    "linear_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_attention_heads": 64,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 256,
    "qk_rope_head_dim": 0,
    "v_head_dim": 256,
    "index_topk": 2048,
    "index_kpool": 4,
    "hc_mult": 4,
    "n_routed_experts": 288,
    "num_experts_per_tok": 8,
    "n_group": 1,
    "topk_group": 1,
    "scoring_func": "sigmoid",
    "moe_intermediate_size": 2048,
}


def _unsupported_deployment(vllm_config) -> list[str]:
    """Deployment restrictions common to all specialized decode policies."""
    import torch

    from vllm.config import CompilationMode, CUDAGraphMode
    from vllm.utils.import_utils import has_cutedsl

    parallel, spec = vllm_config.parallel_config, vllm_config.speculative_config
    model = vllm_config.model_config
    quant = vllm_config.quant_config
    text = model.hf_text_config
    geometry = [
        f"{name}={getattr(text, name, None)} (expects {value})"
        for name, value in GEOMETRY.items()
        if getattr(text, name, None) != value
    ]
    compilation = vllm_config.compilation_config
    graphs = compilation.cudagraph_mode
    checks = (
        (not has_cutedsl(), "missing CuTe DSL"),
        (model.dtype != torch.bfloat16, "activation dtypes other than BF16"),
        (
            model.quantization != "fp8"
            or quant is None
            or not quant.is_checkpoint_fp8_serialized
            or quant.weight_block_size != [128, 128],
            "checkpoints other than native block-FP8 (128x128)",
        ),
        (bool(geometry), "this model geometry: " + ", ".join(geometry)),
        (not getattr(text, "mhc", False), "checkpoints without mHC"),
        (parallel.tensor_parallel_size != 4, "TP other than 4"),
        (parallel.pipeline_parallel_size != 1, "pipeline parallelism"),
        (parallel.data_parallel_size != 1, "data parallelism"),
        (parallel.enable_expert_parallel, "expert parallelism"),
        (parallel.use_sequence_parallel_moe, "sequence-parallel MoE"),
        (parallel.use_ubatching, "DBO"),
        (parallel.decode_context_parallel_size != 1, "decode context parallel"),
        (parallel.prefill_context_parallel_size != 1, "prefill context parallel"),
        (vllm_config.model_config.enable_sleep_mode, "sleep mode"),
        (vllm_config.lora_config is not None, "LoRA"),
        # The decode loop runs in forward; a compiled forward would trace it.
        (compilation.mode == CompilationMode.STOCK_TORCH_COMPILE, "torch.compile"),
        # The loop runs only in FULL decode graphs; eager and PIECEWISE steps run stock.
        (
            graphs is None or graphs.decode_mode() != CUDAGraphMode.FULL,
            "deployments without FULL decode CUDA graphs (e.g. enforce-eager)",
        ),
        (spec is not None and spec.method != "mtp", "speculative methods but MTP"),
        (
            spec is not None and spec.num_speculative_tokens > 7,
            "more than 7 speculative tokens",
        ),
        # The KDA decode reads its projection weights directly as BF16.
        (
            getattr(quant, "online_quantization_config", None) is not None,
            "online quantization (--quantization-config)",
        ),
    )
    return [reason for failed, reason in checks if failed]
