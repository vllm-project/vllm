# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark specialized decode: a decode executor over the stock backbone, enabled
by ``additional_config={"decode_backend": "latency_specialized"}``."""

from dataclasses import dataclass

# The one text config the providers are built for.
MUSE_SPARK_CONFIG = {
    "hidden_size": 8192,
    "num_attention_heads": 128,
    "num_key_value_heads": 16,
    "head_dim": 64,
    "num_local_experts": 256,
    "num_experts_per_tok": 8,
    "moe_hidden_size": 4096,
    "num_shared_experts": 0,
}
TP_SIZES = (4, 8, 16)


@dataclass(frozen=True)
class DecodeConfig:
    tp: int
    # Tokens per request in a verify step (1 + draft tokens).
    q: int
    max_context: int
    capture_sizes: tuple[int, ...]


def decode_backend(vllm_config) -> str:
    """The requested ``decode_backend`` (``additional_config``), "auto" if unset."""
    extra = vllm_config.additional_config
    return extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"


def specialized_decode_config(vllm_config) -> DecodeConfig | None:
    """The deployment's specialized decode settings, None for the stock decode.
    Raises when the specialized decode is requested for an unsupported deployment."""
    backend = decode_backend(vllm_config)
    if backend == "auto":
        return None
    if backend != "latency_specialized":
        raise ValueError("Supported decode_backend values are auto/latency_specialized")
    reason = unsupported(vllm_config)
    if reason is not None:
        raise ValueError(f"Muse Spark specialized decode does not support {reason}")
    return DecodeConfig(
        tp=vllm_config.parallel_config.tensor_parallel_size,
        q=1 + vllm_config.speculative_config.num_speculative_tokens,
        max_context=vllm_config.model_config.max_model_len,
        capture_sizes=tuple(
            vllm_config.compilation_config.cudagraph_capture_sizes or ()
        ),
    )


def unsupported(vllm_config) -> str | None:
    """The first reason this deployment cannot run the specialized decode, or None.

    Every line reads the config only, so every rank gets the same answer. The
    providers' own requirements (attention backend, context length, all-reduce
    backend) follow the deployment lines.
    """
    import torch

    from vllm import envs
    from vllm.config import CompilationMode, CUDAGraphMode
    from vllm.platforms import current_platform

    from .decode import provider_requirements

    parallel, spec = vllm_config.parallel_config, vllm_config.speculative_config
    model, cache = vllm_config.model_config, vllm_config.cache_config
    compilation = vllm_config.compilation_config
    graphs = compilation.cudagraph_mode
    tp = parallel.tensor_parallel_size
    kv_dtype = model.dtype if cache.cache_dtype == "auto" else cache.cache_dtype
    draft_model = None if spec is None else spec.draft_model_config
    draft = None if draft_model is None else draft_model.hf_config
    checks = (
        (not current_platform.is_device_capability_family(100), "this GPU (SM100/103)"),
        (
            any(
                getattr(model.hf_text_config, name, None) != value
                for name, value in MUSE_SPARK_CONFIG.items()
            ),
            "text configs but Muse Spark Open's",
        ),
        (tp not in TP_SIZES, "TP other than 4/8/16"),
        (parallel.pipeline_parallel_size != 1, "pipeline parallelism"),
        (
            not parallel.enable_expert_parallel
            or parallel.data_parallel_size * tp != tp,
            "expert parallelism other than EP = TP",
        ),
        (parallel.use_ubatching, "DBO"),
        (parallel.decode_context_parallel_size != 1, "decode context parallel"),
        (model.enable_sleep_mode, "sleep mode"),
        (vllm_config.lora_config is not None, "LoRA"),
        (
            vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe",
            "the MegaMoE backend",
        ),
        # The runner's ModelState hooks build the executor and bind the KV caches.
        (not vllm_config.use_v2_model_runner, "the V1 model runner"),
        # The decode loop runs in forward; a compiled forward would trace it.
        # The DSpark draft's fused layers are exact to its eager layers only,
        # and a compiled draft would trace their call checks.
        (
            compilation.mode != CompilationMode.NONE,
            "torch.compile (compilation modes but NONE)",
        ),
        # The loop runs only in FULL decode graphs; eager and PIECEWISE steps run stock.
        (
            graphs is None or graphs.decode_mode() != CUDAGraphMode.FULL,
            "deployments without FULL decode CUDA graphs (e.g. enforce-eager)",
        ),
        (
            spec is None or spec.method != "dspark",
            "speculative methods but DSpark",
        ),
        (
            spec is not None and spec.enable_adaptive_verification,
            "adaptive verification",
        ),
        # Verify steps are one request of 8 tokens.
        (
            spec is not None and spec.num_speculative_tokens != 7,
            "speculative token counts but 7",
        ),
        # The draft runs the fused Markov loop (markov.py), which the draft
        # model checks again at load against its weights.
        (
            spec is not None and spec.draft_sample_method != "probabilistic",
            "draft sampling methods but probabilistic",
        ),
        (
            spec is not None and model.head_dtype != torch.bfloat16,
            "draft logits but BF16 (head_dtype)",
        ),
        (
            spec is not None and model.use_fp64_gumbel,
            "FP64 Gumbel sampling",
        ),
        (getattr(vllm_config, "watermark_config", None) is not None, "watermarking"),
        (getattr(draft, "dspark_draft_topk", None) is not None, "a DSpark draft top-k"),
        (
            not getattr(draft, "sample_from_anchor", True),
            "DSpark drafts without sample_from_anchor",
        ),
        (
            getattr(draft, "draft_vocab_size", None)
            not in (None, getattr(draft, "vocab_size", None)),
            "a reduced DSpark draft vocabulary",
        ),
        (kv_dtype not in ("bfloat16", torch.bfloat16), "KV cache dtypes but BF16"),
        (cache.block_size != 16, "KV cache block sizes but 16"),
        (vllm_config.kv_transfer_config is not None, "KV transfer"),
        (envs.VLLM_BATCH_INVARIANT, "batch-invariant mode"),
    )
    reason = next((reason for failed, reason in checks if failed), None)
    return reason if reason is not None else provider_requirements(vllm_config)
