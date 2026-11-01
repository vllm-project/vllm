# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash specialized decode: a decode executor over the stock backbone for
GB300 TP4, enabled by ``additional_config={"decode_backend": "latency_specialized"}``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeConfig:
    max_context: int
    # Tokens per request in a verify step (1 + draft tokens; 1 without speculation).
    q: int
    capture_sizes: tuple[int, ...]
    # Overrides of the per-token-count dispatch (see ``policy.DEFAULT``).
    policy: tuple[tuple[str, str], ...] = ()


def specialized_decode_config(vllm_config) -> DecodeConfig | None:
    """The deployment's specialized decode settings, None for the stock decode.
    Raises when the specialized decode is requested for an unsupported deployment."""
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    if backend == "auto":
        return None
    if backend != "latency_specialized":
        raise ValueError("Supported decode_backend values are auto/latency_specialized")
    reasons = unsupported(vllm_config)
    if reasons:
        raise ValueError(
            "GLM-5.3-Flash specialized decode does not support " + "; ".join(reasons)
        )
    spec = vllm_config.speculative_config
    return DecodeConfig(
        max_context=vllm_config.model_config.max_model_len,
        q=1 + spec.num_speculative_tokens if spec is not None else 1,
        capture_sizes=tuple(
            vllm_config.compilation_config.cudagraph_capture_sizes or ()
        ),
        policy=tuple(extra.get("policy", {}).items()),
    )


def kda_dtypes(vllm_config):
    """(conv, recurrent) KDA state dtypes the deployment's cache config selects."""
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator

    cache = vllm_config.cache_config
    return MambaStateDtypeCalculator.kda_state_dtype(
        vllm_config.model_config.dtype,
        cache.mamba_cache_dtype,
        cache.mamba_ssm_cache_dtype,
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


def unsupported(vllm_config) -> list[str]:
    """Every reason this deployment cannot run the specialized decode."""
    import torch

    from vllm.config import CompilationMode, CUDAGraphMode
    from vllm.platforms import current_platform

    parallel, spec = vllm_config.parallel_config, vllm_config.speculative_config
    text = vllm_config.model_config.hf_text_config
    geometry = [
        f"{name}={getattr(text, name, None)} (expects {value})"
        for name, value in GEOMETRY.items()
        if getattr(text, name, None) != value
    ]
    compilation = vllm_config.compilation_config
    graphs = compilation.cudagraph_mode
    checks = (
        (not current_platform.is_device_capability_family(100), "this GPU (SM100/103)"),
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
        # The runner's ModelState hooks build the executor and prepare the KV layouts.
        (not vllm_config.use_v2_model_runner, "the V1 model runner"),
        # The decode loop runs in forward; a compiled forward would trace it.
        (compilation.mode == CompilationMode.STOCK_TORCH_COMPILE, "torch.compile"),
        # The loop runs only in FULL decode graphs; eager and PIECEWISE steps run stock.
        (
            graphs is None or graphs.decode_mode() != CUDAGraphMode.FULL,
            "deployments without FULL decode CUDA graphs (e.g. enforce-eager)",
        ),
        (
            not str(vllm_config.cache_config.cache_dtype).startswith("fp8"),
            "KV cache dtypes other than FP8",
        ),
        (spec is not None and spec.method != "mtp", "speculative methods but MTP"),
        (
            spec is not None and spec.num_speculative_tokens > 7,
            "more than 7 speculative tokens",
        ),
        # The KDA decode keeps a BF16 conv state and an FP32 recurrent state in place;
        # it maintains neither other cache dtypes nor ReplaySSM / RecoverSSM records.
        (
            kda_dtypes(vllm_config) != (torch.bfloat16, torch.float32),
            "KDA cache dtypes other than a BF16 conv state and an FP32 recurrent "
            "state (mamba cache dtype)",
        ),
        (
            vllm_config.cache_config.use_replayssm
            or vllm_config.cache_config.use_kda_recoverssm,
            "ReplaySSM / RecoverSSM (--use-replayssm)",
        ),
    )
    return [reason for failed, reason in checks if failed]
