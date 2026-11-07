# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode: a CuTe decode executor over the stock backbone, enabled
by `additional_config={"decode_backend": "latency_specialized"}`."""

from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)

SPECIALIZED = "latency_specialized"


@dataclass(frozen=True)
class DecodeConfig:
    max_context: int
    dcp: int
    # Tokens per request in a verify step (1 + draft tokens; 1 without speculation).
    q: int
    capture_sizes: tuple[int, ...]
    # The FlashInfer sparse MLA kernels the DSA layers pick from per batch.
    fi_kernels: tuple[str, ...] = ()
    # The row capacities of the prims_ts plans (a batch runs on the smallest that
    # holds it: the plan fixes the split-KV schedule and its BF16 partial roundings).
    prims_ts_rows: tuple[int, ...] = ()
    # Kimi K3.1 (DSA layers next to the MLA ones).
    dsa: bool = False


def specialized_decode_config(vllm_config) -> DecodeConfig | None:
    """The deployment's specialized decode settings, None for the stock decode.
    Raises when the specialized decode is requested for an unsupported deployment."""
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    if backend == "auto":
        return None
    if backend != "latency_specialized":
        raise ValueError("Supported decode_backend values are auto/latency_specialized")
    reason = unsupported(vllm_config)
    if reason is not None:
        raise ValueError(f"Kimi K3 specialized decode does not support {reason}")
    spec = vllm_config.speculative_config
    if spec is not None:
        # The draft model loads after this target: build it as the specialized
        # drafter (its own module, ``drafter.model``).
        from ..drafter.model import register

        register()
    return DecodeConfig(
        max_context=vllm_config.model_config.max_model_len,
        dcp=vllm_config.parallel_config.decode_context_parallel_size,
        q=1 + spec.num_speculative_tokens if spec is not None else 1,
        capture_sizes=tuple(
            vllm_config.compilation_config.cudagraph_capture_sizes or ()
        ),
        fi_kernels=dsa_fi_kernels(vllm_config),
        prims_ts_rows=prims_ts_rows(vllm_config),
        dsa=getattr(vllm_config.model_config.hf_text_config, "dsa_config", None)
        is not None,
    )


def dsa_fi_kernels(vllm_config) -> tuple[str, ...]:
    """The FlashInfer sparse MLA kernels of the DSA layers on a per-tensor FP8 or
    nvfp4_ds_mla cache (none for other models and caches)."""
    text = vllm_config.model_config.hf_text_config
    dtype = vllm_config.cache_config.cache_dtype
    if getattr(text, "dsa_config", None) is None or dtype not in FI_CACHE_DTYPES:
        return ()
    from ..cutedsl.dsa.attn_fi import fi_kernels

    return fi_kernels(dtype)


def prims_ts_rows(vllm_config) -> tuple[int, ...]:
    """The prims_ts plans' row capacities: the deployment's CUDA graph sizes and the
    powers of two up to its widest call (FLASHINFER_MLA_SPARSE's buckets)."""
    # FLASHINFER_MLA_SPARSE does not run prims_ts yet.
    return ()


def kda_dtypes(vllm_config):
    """(conv, recurrent) KDA state dtypes the deployment's cache config selects."""
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator

    cache = vllm_config.cache_config
    return MambaStateDtypeCalculator.kda_state_dtype(
        vllm_config.model_config.dtype,
        cache.mamba_cache_dtype,
        cache.mamba_ssm_cache_dtype,
    )


def unsupported(vllm_config) -> str | None:
    import torch

    kda_allowed = ((torch.bfloat16, torch.float32), (torch.float32, torch.float32))

    from vllm import envs
    from vllm.config import CompilationMode, CUDAGraphMode
    from vllm.platforms import current_platform

    parallel, spec = vllm_config.parallel_config, vllm_config.speculative_config
    text = vllm_config.model_config.hf_text_config
    moe = tuple(
        getattr(text, name, None)
        for name in (
            "hidden_act",
            "activation_situ_beta",
            "activation_situ_linear_beta",
            "num_experts",
            "num_experts_per_token",
            "num_expert_group",
            "topk_group",
            "moe_router_activation_func",
        )
    )
    compilation = vllm_config.compilation_config
    graphs = compilation.cudagraph_mode
    # Kimi K3.1: sparse (DSA) MLA layers over the packed MLA and indexer caches,
    # fp8_ds_mla on FlashMLA sparse or per-tensor FP8 on FlashInfer sparse.
    dsa = getattr(text, "dsa_config", None) is not None
    tp1 = parallel.tensor_parallel_size == 1
    ep = parallel.enable_expert_parallel
    checks = (
        (not current_platform.is_device_capability_family(100), "this GPU (SM100/103)"),
        (
            moe != ("situ", 4.0, 25.0, 896, 16, 1, 1, "sigmoid"),
            "MoE configs but the one-group sigmoid top-16 SiTU MoE of Kimi K3",
        ),
        (
            parallel.tensor_parallel_size not in ((1, 4, 8, 16) if ep else (1, 8, 16)),
            "TP other than 1/8/16 (1/4/8/16 with expert parallelism)",
        ),
        (parallel.pipeline_parallel_size != 1, "pipeline parallelism"),
        # Expert parallel: DEP (TP1 attention per DP rank: runtime.dep) or the hybrid
        # (TP-G attention groups, routed experts EP over every rank: runtime.hybrid),
        # both over MegaMoE; TP8 / TP16 without EP run their TP-sharded MoE providers.
        (
            ep and not tp1 and not dsa,
            "expert parallelism with TP > 1 without DSA layers (Kimi K3 MLA)",
        ),
        (
            ep
            and not tp1
            and vllm_config.kernel_config.moe_backend != "deep_gemm_mega_moe",
            "expert parallelism with TP > 1 on MoE backends other than MegaMoE",
        ),
        # The hybrid step all-reduces o_proj's full-row partial; the fused GEMM +
        # reduce-scatter o_proj returns a row shard instead.
        (
            ep and not tp1 and envs.VLLM_ENABLE_GEMM_RS,
            "expert parallelism with TP > 1 under the GEMM + reduce-scatter o_proj",
        ),
        (
            parallel.tensor_parallel_size == 1 and not dsa,
            "TP1 (DEP) without DSA layers (Kimi K3 MLA)",
        ),
        (parallel.use_ubatching, "DBO"),
        (
            parallel.decode_context_parallel_size
            not in (1, parallel.tensor_parallel_size),
            "decode context parallel other than 1 or TP",
        ),
        (envs.VLLM_BATCH_INVARIANT, "batch invariance"),
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
            spec is not None and spec.method != "dspark",
            "speculative methods but DSpark",
        ),
        (
            spec is not None and spec.enable_adaptive_verification,
            "adaptive verification",
        ),
        # The KDA cores (KdaSplit, KdaWide) keep an FP32 recurrent state and a BF16
        # or FP32 conv state in place (the taps are BF16 projection values: exact
        # in both).
        (
            kda_dtypes(vllm_config) not in kda_allowed,
            (
                "KDA cache dtypes other than a BF16 or FP32 conv state and an FP32 "
                "recurrent state (mamba cache dtype)"
            ),
        ),
        # The verify kernels take at most 8 tokens per request (KdaSplit's barriers).
        (
            spec is not None and spec.num_speculative_tokens > 7,
            "more than 7 speculative tokens",
        ),
        (
            spec is not None and envs.VLLM_KIMI_K3_AUX_ATTN_RES_STREAM,
            "the AttnRes aux stream",
        ),
        # The MLA cores read a per-tensor FP8 cache.
        (
            not dsa and vllm_config.cache_config.cache_dtype not in MLA_CACHE_DTYPES,
            "MLA layers on KV cache dtypes other than per-tensor fp8",
        ),
        # TP1: the split query writes FlashInfer's per-tensor FP8 query (96 heads).
        (
            tp1 and vllm_config.cache_config.cache_dtype not in FI_CACHE_DTYPES,
            "TP1 (DEP) on KV cache dtypes other than per-tensor fp8 or nvfp4_ds_mla",
        ),
        (
            dsa
            and vllm_config.cache_config.cache_dtype
            not in ("fp8_ds_mla", "fp8", "fp8_e4m3", "nvfp4_ds_mla"),
            (
                "DSA layers on KV cache dtypes other than fp8_ds_mla (FlashMLA "
                "sparse), per-tensor fp8 (FlashInfer sparse) or nvfp4_ds_mla "
                "(FlashInfer sparse prims_ts)"
            ),
        ),
    )
    reason = next((reason for failed, reason in checks if failed), None)
    if reason is None and dsa:
        reason = dsa_unsupported(vllm_config)
    return reason


def dsa_unsupported(vllm_config) -> str | None:
    """Why the K3.1 DSA layers' specialized kernels cannot run: the query and index
    fronts implement the K3.1 indexer (no indexer RoPE, LayerNorm keys; an FP8
    wq_index is refused when the layer is built), and on a FlashInfer sparse cache
    the per-batch kernel policy needs FlashInfer's prims_ts kernel."""
    dsa = vllm_config.model_config.hf_text_config.dsa_config
    if getattr(dsa, "index_rope_dim", 0) != 0:
        return "DSA indexer RoPE (index_rope_dim > 0)"
    if getattr(dsa, "index_k_norm_type", None) != "layernorm":
        return "DSA indexer keys normalized other than by LayerNorm"
    dtype = vllm_config.cache_config.cache_dtype
    if dtype in FI_CACHE_DTYPES:
        return f"DSA layers on a {dtype} cache without FlashInfer prims_ts"
    return None


# The per-tensor FP8 cache dtypes the MLA cores read; the DSA caches the FlashInfer
# sparse MLA kernels read.
MLA_CACHE_DTYPES = ("fp8", "fp8_e4m3")
FI_CACHE_DTYPES = ("fp8", "fp8_e4m3", "nvfp4_ds_mla")
