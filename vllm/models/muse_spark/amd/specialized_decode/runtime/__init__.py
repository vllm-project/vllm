# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark specialized decode on MI355X (layer mode), enabled by
``--monokernel`` or ``additional_config={"decode_backend": "latency_specialized"}``.

``specialized_decode_config`` runs in the model's ``__init__``, before the weights
load. It returns None for the stock decode, and raises with every reason from
``refuse_config`` when the specialized decode is requested for a deployment it
cannot serve. Checks that need the loaded weights run when the executor is built
(``decode.SpecializedDecode``, ``guards.refuse_model``), those that need the KV
cache in ``prepare_kv`` (``guards.refuse_caches``).

Nothing here imports FlyDSL or the model: the model imports this module."""

from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    BLOCKS,
    MAX_ROWS,
    MUSE_SHAPE_FIELDS,
    MUSE_SHAPES,
    TP,
)

logger = init_logger(__name__)

AUTO, SPECIALIZED = "auto", "latency_specialized"
HINT = "decode_backend=auto without --monokernel runs vLLM's decode"
KV_BLOCK = 16  # tokens a KV block (--block-size 16)
REDUCE_DTYPES = ("fp32", "bf16")
OFF_ROCM = "this platform (ROCm only)"


@dataclass(frozen=True)
class DecodeConfig:
    # rows a request decodes: 8 under DSpark (K = 7), else 1
    q: int
    # the rows a launch serves: whole requests, at most MAX_ROWS rows, no more
    # than the scheduler can ever batch (max_num_seqs requests); one build each
    widths: tuple[int, ...]


def decode_widths(q: int, max_num_seqs: int | None) -> tuple[int, ...]:
    """Every uniform decode width of whole requests up to MAX_ROWS rows that the
    scheduler can produce: an eager step of one of them never pads."""
    cap = MAX_ROWS if max_num_seqs is None else max_num_seqs * q
    return tuple(w for w in range(q, MAX_ROWS + 1, q) if w <= cap)


def moe_mode(vllm_config) -> str:
    """The experts' parallelism K2 is built for: "ep" (TEP8: attention TP8,
    this rank's 32 experts whole) under ``--enable-expert-parallel``, else "tp"."""
    return "ep" if vllm_config.parallel_config.enable_expert_parallel else "tp"


def decode_backend(vllm_config) -> str:
    """``additional_config["decode_backend"]``, "auto" when absent or when
    ``additional_config`` is not a dict."""
    extra = vllm_config.additional_config
    return extra.get("decode_backend", AUTO) if isinstance(extra, dict) else AUTO


def requested(vllm_config) -> bool:
    """The one place that reads the switch: ``--monokernel``
    (``kernel_config.monokernel``) or ``decode_backend=latency_specialized``.
    Any other ``decode_backend`` is an error, with or without ``--monokernel``."""
    backend = decode_backend(vllm_config)
    if backend not in (AUTO, SPECIALIZED):
        raise ValueError(
            f"decode_backend must be '{AUTO}' or '{SPECIALIZED}', got {backend!r}"
        )
    kernel_config = getattr(vllm_config, "kernel_config", None)
    return backend == SPECIALIZED or bool(getattr(kernel_config, "monokernel", False))


def _refuse(reasons: list[str]) -> None:
    raise ValueError(
        "Muse Spark specialized decode (--monokernel / decode_backend="
        "latency_specialized) does not support: " + "; ".join(reasons) + f" ({HINT})"
    )


def specialized_decode_config(vllm_config) -> DecodeConfig | None:
    """The deployment's specialized decode settings, None for the stock decode
    (nothing is checked then). Raises when it is requested for an unsupported
    deployment, listing every reason."""
    if not requested(vllm_config):
        return None
    reasons = refuse_config(vllm_config)
    if reasons:
        _refuse(reasons)
    if eager_decode(vllm_config):
        logger.warning_once(
            "Muse Spark specialized decode: decode steps run eagerly "
            "(cudagraph_mode NONE); use FULL_AND_PIECEWISE or FULL_DECODE_ONLY "
            "for serving"
        )
    spec = vllm_config.speculative_config
    q = 1 if spec is None else spec.num_speculative_tokens + 1
    max_num_seqs = getattr(
        getattr(vllm_config, "scheduler_config", None), "max_num_seqs", None
    )
    return DecodeConfig(q=q, widths=decode_widths(q, max_num_seqs))


def refuse_config(vllm_config) -> list[str]:
    """Stage 1: every reason this deployment cannot run the specialized decode
    (empty: it can). Config only: the loaded weights and the KV caches are checked
    later. Default-deny: the kernels serve none of the features below."""
    from vllm import envs
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import CompilationMode, CUDAGraphMode
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    mc, pc, cc = (
        vllm_config.model_config,
        vllm_config.parallel_config,
        vllm_config.cache_config,
    )
    spec, oc = vllm_config.speculative_config, vllm_config.offload_config
    compilation = vllm_config.compilation_config
    cudagraph_mode = compilation.cudagraph_mode
    text = mc.hf_text_config
    reduce_dtype = envs.VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE
    quant = vllm_config.quant_config
    checks = (
        (pc.tensor_parallel_size != TP, f"TP {pc.tensor_parallel_size} (TP{TP} only)"),
        (pc.pipeline_parallel_size != 1, "pipeline parallelism"),
        (pc.data_parallel_size != 1, "data parallelism"),
        # TEP8 (moe_mode "ep"): attention TP8, rank r's experts [32 r, 32 r + 32)
        (
            pc.enable_expert_parallel
            and getattr(pc, "expert_placement_strategy", "linear") != "linear",
            "expert parallelism with a non-linear expert placement",
        ),
        (pc.enable_eplb, "EPLB"),
        (pc.decode_context_parallel_size != 1, "decode context parallelism"),
        (pc.prefill_context_parallel_size != 1, "prefill context parallelism"),
        # the kernels read one batch's metadata a step
        (pc.use_ubatching, "DBO (dual-batch overlap)"),
        (pc.use_sequence_parallel_moe, "sequence-parallel MoE"),
        (vllm_config.lora_config is not None, "LoRA"),
        # --kv-offloading-size creates a KVTransferConfig too
        (
            vllm_config.kv_transfer_config is not None,
            "KV transfer / KV offloading connectors",
        ),
        (
            vllm_config.aux_output_config.enable_return_routed_experts,
            "routed-expert capture",
        ),
        # the kernels read the loaded weights in place
        (mc.enable_sleep_mode, "sleep mode"),
        (
            oc.uva.cpu_offload_gb > 0 or oc.prefetch.offload_group_size > 0,
            "CPU weight offload",
        ),
        (vllm_config.weight_transfer_config is not None, "weight updates"),
        (mc.enable_prompt_embeds, "prompt embeds"),
        # the kernels serve plain decode or a DSpark verify of whole requests
        (
            spec is not None and spec.method != "dspark",
            f"speculative method {getattr(spec, 'method', None)!r} (DSpark only)",
        ),
        (
            spec is not None and spec.num_speculative_tokens + 1 not in (1, MAX_ROWS),
            f"num_speculative_tokens {getattr(spec, 'num_speculative_tokens', None)}"
            f" (0 or {MAX_ROWS - 1} only)",
        ),
        (
            spec is not None and spec.enable_adaptive_verification,
            "adaptive verification",
        ),
        (
            spec is not None and spec.uses_dynamic_speculative_decoding(),
            "dynamic speculative decoding",
        ),
        # the model state hooks build the executor and bind it to each KV cache
        (
            not vllm_config.use_v2_model_runner,
            "the V1 model runner (set VLLM_USE_V2_MODEL_RUNNER=1)",
        ),
        # plan() and the layer kernels run in forward; a compiled forward would
        # trace them
        (
            compilation.mode == CompilationMode.STOCK_TORCH_COMPILE,
            "torch.compile (compilation mode STOCK_TORCH_COMPILE)",
        ),
        # a PIECEWISE decode graph replays the stock layers only
        (
            cudagraph_mode is not None
            and cudagraph_mode.decode_mode() == CUDAGraphMode.PIECEWISE,
            "PIECEWISE decode CUDA graphs (use FULL_AND_PIECEWISE or FULL_DECODE_ONLY)",
        ),
        (mc.dtype != torch.bfloat16, f"model dtype {mc.dtype} (bfloat16 only)"),
        (
            cc.cache_dtype not in ("auto", "bfloat16"),
            f"KV cache dtype {cc.cache_dtype!r} (bf16 KV only)",
        ),
        (
            cc.block_size is not None and cc.block_size != KV_BLOCK,
            f"KV block size {cc.block_size} (--block-size {KV_BLOCK} only)",
        ),
        # the KV cache is written by the model (fused QK-norm/RoPE/KV insert)
        (
            rocm_aiter_ops.is_shuffle_kv_cache_enabled(),
            "the AITER shuffled KV cache layout",
        ),
        (
            reduce_dtype not in REDUCE_DTYPES,
            f"VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE={reduce_dtype!r} (fp32 or bf16)",
        ),
        (
            not isinstance(quant, QuarkConfig),
            f"quantization {'none' if quant is None else quant.get_name()!r} "
            "(the Quark OCP-MXFP4 checkpoint only)",
        ),
        # pad rows: the kernels read forward_context.is_padding
        (
            not envs.VLLM_MOE_SKIP_PADDING,
            "VLLM_MOE_SKIP_PADDING=0 (the specialized decode reads the per-step "
            "pad mask forward_context.is_padding, which MRV2 maintains only with "
            "VLLM_MOE_SKIP_PADDING=1)",
        ),
        # the fail-stop poll runs in compute_logits
        (
            bool(pc.enable_batch_sharded_sampling),
            "batch-sharded sampling (the fail-stop poll runs in compute_logits; "
            "MRV2 calls compute_logits_local instead)",
        ),
    )
    reasons = [why for bad, why in checks if bad]
    reasons += _attention_reasons(vllm_config.attention_config, AttentionBackendEnum)
    shapes = text_shapes(text)
    if shapes != MUSE_SHAPES:
        reasons.append(
            f"model shapes {shapes} (hidden, heads, kv heads, head dim, experts, "
            "top-k, moe hidden, moe intermediate, shared experts, sliding window; "
            f"Muse Spark Open is {MUSE_SHAPES})"
        )
    if getattr(getattr(text, "gated_delta_net", None), "enable", False):
        reasons.append("gated_delta_net layers")
    return reasons + pattern_reasons(text) + platform_reasons()


unsupported = refuse_config  # the RFC's name


def _attention_reasons(attention_config, backends) -> list[str]:
    """ROCM_AITER_FA for every layer (the fused KV insert and the metadata
    ``plan()`` reads), set explicitly: automatic selection may pick another."""
    want = backends.ROCM_AITER_FA
    backend = getattr(attention_config, "backend", None)
    bad = [] if backend == want else ["auto" if backend is None else backend.name]
    per_kind = getattr(attention_config, "backend_per_kind", None) or {}
    bad += [f"{kind}={b.name}" for kind, b in per_kind.items() if b != want]
    if not bad:
        return []
    return [
        f"attention backend {', '.join(bad)} (ROCM_AITER_FA only: "
        "--attention-backend ROCM_AITER_FA)"
    ]


def text_shapes(text) -> tuple:
    """The text config's shapes the kernels are written for, in the order of
    ``spec.MUSE_SHAPES``."""
    return tuple(getattr(text, name, None) for name in MUSE_SHAPE_FIELDS)


def pattern_reasons(text) -> list[str]:
    """The layer pattern: every MLP sparse, sliding or full attention only,
    RoPE exactly on the sliding layers."""
    n = getattr(text, "num_hidden_layers", None)
    mlp = getattr(text, "mlp_layer_types", None)
    kinds = getattr(text, "layer_types", None)
    thetas = getattr(text, "layer_rope_theta", None)
    if (
        n is None
        or mlp is None
        or kinds is None
        or thetas is None
        or any(len(v) != n for v in (mlp, kinds, thetas))
    ):
        return ["a layer pattern (layer_types, mlp_layer_types, layer_rope_theta)"]
    reasons = []
    if any(t != "sparse" for t in mlp):
        reasons.append("dense MLP layers")
    reasons += [
        f"layer type {t!r}"
        for t in dict.fromkeys(kinds)
        if t not in ("sliding_attention", "full_attention")
    ]
    if any((th == 0) != (t == "full_attention") for th, t in zip(thetas, kinds)):
        reasons.append("RoPE on full-attention layers or NoPE on sliding layers")
    return reasons


def platform_reasons() -> list[str]:
    """The first platform requirement this process misses, if any."""
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return [OFF_ROCM]
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return ["this GPU (gfx950 / MI355X only)"]
    try:
        from vllm.models.muse_spark.amd.specialized_decode.mono.common import (  # noqa: F401
            ops,
        )
    except Exception as err:  # a refusal reason, never a failed model init
        return [f"FlyDSL and AITER's FlyDSL helpers ({err})"]
    try:
        cus = current_platform.num_compute_units(
            torch.accelerator.current_device_index()
        )
    except Exception as err:
        return [f"the GPU's CU count ({err})"]
    if cus < BLOCKS:
        # every CTA of a launch must be resident: a partitioned GPU deadlocks
        return [f"{cus} CUs (the kernels need {BLOCKS} resident CTAs, one a CU)"]
    return []


def eager_decode(vllm_config) -> bool:
    """Decode steps run eagerly (cudagraph_mode NONE): served, but slow."""
    from vllm.config import CUDAGraphMode

    mode = vllm_config.compilation_config.cudagraph_mode
    return mode is None or mode == CUDAGraphMode.NONE
