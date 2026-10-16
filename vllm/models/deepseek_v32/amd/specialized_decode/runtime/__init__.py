# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.2 specialized decode: one FlyDSL MonoKernel per MoE decoder layer inside the
stock layer loop (layer mode), enabled by
``additional_config={"decode_backend": "latency_specialized"}``. Options (the
``DecodeConfig`` fields) go under ``additional_config["specialized_decode"]``.

``specialized_decode_config`` runs in the model's ``__init__``, before the weights
load. It returns None for the stock decode, and raises with every reason from
``unsupported`` when the specialized decode is requested for a deployment it cannot
serve. Checks that need the loaded weights run when the executor is built
(``decode.SpecializedDecode``), those that need the KV cache in ``prepare_kv``."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.config import (
    GLM5_KERNEL_SAMPLES,
)

logger = init_logger(__name__)

AUTO, SPECIALIZED = "auto", "latency_specialized"
# additional_config key of the DecodeConfig options
OPTIONS = "specialized_decode"
# batch sizes the GLM kernel can be built for (at most 12 requests per step at TP8)
BATCH_SIZES = GLM5_KERNEL_SAMPLES
# largest decode batch the kernels serve; larger batches run vLLM's decode. Above 8
# the kernel switches to a second row group: TPOT vs vLLM is 1.50x at batch 8, 1.24x
# at 10 and 1.06x at 12, and throughput peaks at 8 (MI355X, BF16, 3k context)
MAX_BATCH_SIZE = 8


@dataclass(frozen=True)
class DecodeConfig:
    """Step and kernel options. ``decode_config`` fills it from the defaults, the
    ``additional_config["specialized_decode"]`` options and vLLM's max_model_len."""

    layers: tuple[int, ...] = tuple(range(3, 78))
    poll_limit: int = 20_000_000  # ~4 s per mailbox wait
    # indexer layers refresh the top-k through vLLM's attention ("attn") or the
    # indexer alone ("indexer_only")
    indexer_mode: str = "indexer_only"
    # eager only: rank-uniform state vote (device sync + CPU all-reduce)
    step_sync: bool = True
    check_every: int = 1  # eager health check every N kernel steps (0 = never)
    max_model_len: int = 4096  # RoPE table rows
    enabled: bool = True  # initial state
    poll_early_out: bool = True  # after one expired poll, skip the step's other waits
    indexer_trim: bool = True  # indexer_only: skip work whose results are discarded
    cache_hoist: bool = True  # cache stage: positions / slots loaded ahead of its wait
    split_keys64: bool = True  # 64-key sparse-MLA split tasks (batch sizes <= 8 only)
    # debug: FP8-attention checkpoints run the non-fused indexer path (A/B checks)
    debug_no_fused_indexer: bool = False
    # captured isfinite count of each step's output (+ fail-stop on it)
    device_nonfinite: bool = False
    failstop_nonfinite: bool = False
    # expired kernel polls: "raise" (fail-stop), "warn" (log and continue) or "off" (no
    # fail-stop watch); with "warn" / "off" the eager health check (check_every) still
    # disables the kernels after an expiry
    failstop: str = "raise"

    def __post_init__(self):
        for name, allowed in (
            ("indexer_mode", ("attn", "indexer_only")),
            ("failstop", ("raise", "warn", "off")),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        """The batch sizes the kernels are built for; a smaller batch pads up to the
        next one."""
        return tuple(b for b in BATCH_SIZES if b <= MAX_BATCH_SIZE)


def decode_backend(vllm_config) -> str:
    """The requested decode backend: the one place that reads the switch."""
    extra = vllm_config.additional_config
    return extra.get("decode_backend", AUTO) if isinstance(extra, dict) else AUTO


def specialized_decode_config(vllm_config) -> DecodeConfig | None:
    """The deployment's specialized decode settings, None for the stock decode.
    Raises when the specialized decode is requested for an unsupported deployment,
    listing every reason."""
    backend = decode_backend(vllm_config)
    if backend == AUTO:
        return None
    if backend != SPECIALIZED:
        raise ValueError(
            f"decode_backend must be '{AUTO}' or '{SPECIALIZED}', got {backend!r}"
        )
    reasons, cfg = _reasons_and_config(vllm_config)
    if reasons:
        raise ValueError(
            "GLM-5.2 specialized decode (decode_backend=latency_specialized) does not "
            "support: " + "; ".join(reasons) + f" (decode_backend={AUTO} runs vLLM's "
            "decode)"
        )
    if piecewise_graphs_only(vllm_config):
        logger.warning(
            "GLM-5.2 specialized decode: PIECEWISE-only breakable cudagraphs capture "
            "every decode step as stock; use FULL decode graphs or eager mode"
        )
    assert cfg is not None
    return cfg


def decode_config(vllm_config) -> DecodeConfig:
    """The defaults, overridden by ``additional_config["specialized_decode"]``. The
    RoPE table covers vLLM's max_model_len; step_sync (a device sync and a CPU
    all-reduce per step) defaults off under FULL graphs."""
    over = dict(vllm_config.additional_config.get(OPTIONS) or {})
    if "layers" in over:
        over["layers"] = tuple(over["layers"])
    over.setdefault("step_sync", not full_cudagraphs(vllm_config))
    mml = int(vllm_config.model_config.max_model_len)
    if over.setdefault("max_model_len", mml) < mml:
        logger.warning(
            "GLM-5.2 specialized decode: raising the RoPE table length %d -> vLLM "
            "max_model_len %d",
            over["max_model_len"],
            mml,
        )
        over["max_model_len"] = mml
    return DecodeConfig(**over)


def unsupported(vllm_config) -> list[str]:
    """Every reason this deployment cannot run the specialized decode (empty: it
    can). Config only: the loaded weights are checked when the executor is built."""
    return _reasons_and_config(vllm_config)[0]


def _reasons_and_config(vllm_config) -> tuple[list[str], DecodeConfig | None]:
    """``unsupported``'s reasons and the ``DecodeConfig`` (None if the options are
    invalid), so that ``decode_config`` runs once."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import CompilationMode

    mc, pc = vllm_config.model_config, vllm_config.parallel_config
    model_type = getattr(mc.hf_config, "model_type", None)
    kv_dtype = vllm_config.cache_config.cache_dtype
    checks = (
        (model_type != "glm_moe_dsa", f"model_type {model_type} (GLM-5.2 only)"),
        (pc.tensor_parallel_size != 8, f"TP {pc.tensor_parallel_size} (TP8 only)"),
        (pc.pipeline_parallel_size != 1, "pipeline parallelism"),
        (pc.data_parallel_size != 1, "data parallelism"),
        (pc.enable_expert_parallel, "expert parallelism"),
        (pc.decode_context_parallel_size != 1, "decode context parallelism"),
        (pc.prefill_context_parallel_size != 1, "prefill context parallelism"),
        # the kernels read one batch's attention metadata per step
        (pc.use_ubatching, "DBO (dual-batch overlap)"),
        (vllm_config.speculative_config is not None, "speculative decoding"),
        # the kernels read the loaded weights in place; a wake-up reloads them
        (mc.enable_sleep_mode, "sleep mode"),
        # the kernels replace vLLM's linear and attention layers
        (vllm_config.lora_config is not None, "LoRA"),
        (vllm_config.kv_transfer_config is not None, "KV transfer connectors"),
        (
            vllm_config.aux_output_config.enable_return_routed_experts,
            "routed-expert capture",
        ),
        # the kernels read the routed and shared experts' own tensors
        (
            bool(rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()),
            "fused shared experts (VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1)",
        ),
        (mc.dtype != torch.bfloat16, f"dtype {mc.dtype} (bfloat16 only)"),
        (
            kv_dtype not in ("auto", "bfloat16"),
            f"kv_cache_dtype {kv_dtype} (bfloat16 only)",
        ),
        # the model state hooks build the executor and bind it to each KV cache
        (
            not vllm_config.use_v2_model_runner,
            "the V1 model runner (set VLLM_USE_V2_MODEL_RUNNER=1)",
        ),
        # plan() and the per-layer kernels run in forward; a compiled forward would
        # trace them (DeepseekV32Model has no @support_torch_compile)
        (
            vllm_config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE,
            "torch.compile (compilation mode STOCK_TORCH_COMPILE)",
        ),
    )
    reasons = [why for bad, why in checks if bad]
    why = _platform_reason()
    if why is not None:
        reasons.append(why)
    try:
        cfg = decode_config(vllm_config)
    except (TypeError, ValueError) as e:
        return reasons + [f'additional_config["{OPTIONS}"]: {e}'], None
    reasons += attention_quant_reasons(vllm_config, cfg.layers)
    if cfg.step_sync and full_cudagraphs(vllm_config):
        reasons.append(
            "step_sync=True under FULL cudagraphs (its device sync and CPU all-reduce "
            "cannot run inside a graph capture)"
        )
    return reasons, cfg


# the attention linears as the checkpoint names them (fused_qkv_a_proj = q_a | kv_a)
ATTENTION_LINEARS = (
    "q_a_proj",
    "kv_a_proj_with_mqa",
    "q_b_proj",
    "kv_b_proj",
    "o_proj",
    "indexer.wq_b",
)


def attention_quant_reasons(vllm_config, layers) -> list[str]:
    """Why the mono layers' attention linears cannot be read as checkpointed: each must
    be BF16 (excluded from quantization) or Quark per-channel FP8 with per-token
    dynamic activations (PTPC), which the kernel reads with its per-row scales, and the
    attention proper (all but ``indexer.wq_b``) one precision on every mono layer.
    Block / per-tensor FP8 and non-Quark quantization (e.g. zai-org/GLM-5.2-FP8) are
    refused: the specialized decode never requantizes weights."""
    qc = getattr(vllm_config, "quant_config", None)
    if qc is None:
        return []
    if qc.get_name() != "quark":
        name = qc.get_name()
        return [f"quantization {name} (Quark with BF16 or per-channel FP8 attention)"]
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
        kFp8StaticChannelSym,
    )

    bad: dict[str, set[int]] = {}
    # (layer, linear) -> FP8 of the attention proper's linears
    fp8: dict[int, dict[str, bool]] = {}
    for L in layers:
        for proj in ATTENTION_LINEARS:
            prefix = f"model.layers.{L}.self_attn.{proj}"
            try:
                wk, ak, method = qc.get_quant_method_target(prefix, LinearBase)
            except (NotImplementedError, ValueError) as e:
                why = f"{proj}: {e}"
            else:
                is_bf16 = method is UnquantizedLinearMethod
                if is_bf16 or (
                    wk == kFp8StaticChannelSym and ak == kFp8DynamicTokenSym
                ):
                    if proj != "indexer.wq_b":
                        fp8.setdefault(L, {})[proj] = not is_bf16
                    continue
                why = f"{proj} quantized as {wk} / {ak}"
            bad.setdefault(why, set()).add(L)
    reasons = [
        f"{why} on layers {sorted(Ls)[0]}..{sorted(Ls)[-1]} (attention must be BF16 or "
        "per-channel FP8 with per-token activations)"
        for why, Ls in bad.items()
    ]
    partial = sorted(L for L, d in fp8.items() if len(set(d.values())) > 1)
    if partial:
        d = fp8[partial[0]]
        reasons.append(
            f"partial FP8 attention on layers {partial[0]}..{partial[-1]} (layer "
            f"{partial[0]}: FP8 {[p for p, v in d.items() if v]}, BF16 "
            f"{[p for p, v in d.items() if not v]}; a layer's attention is read all "
            "BF16 or all FP8)"
        )
    elif len({next(iter(d.values())) for d in fp8.values() if d}) > 1:
        on = sorted(L for L, d in fp8.items() if any(d.values()))
        reasons.append(
            f"mixed attention precision across mono layers (FP8 on {len(on)} layers, "
            f"{on[0]}..{on[-1]}; BF16 on the others)"
        )
    return reasons


def _platform_reason() -> str | None:
    try:
        from vllm.platforms.rocm import on_gfx950

        return None if on_gfx950() else "this GPU (gfx950 / MI350X only)"
    except Exception as e:
        return f"this platform (gfx950 check failed: {e!r})"


def full_cudagraphs(vc) -> bool:
    """FULL (whole-forward) cudagraphs: replays run no Python, so the step decision
    and every flag it reads are baked in at capture."""
    mode = vc.compilation_config.cudagraph_mode
    return mode is not None and bool(mode.has_full_cudagraphs())


def piecewise_graphs_only(vc) -> bool:
    """Decode steps captured as breakable PIECEWISE graphs only: plan() runs them
    stock."""
    mode = vc.compilation_config.cudagraph_mode
    if mode is None or full_cudagraphs(vc) or mode.name == "NONE":
        return False
    from vllm import envs as vllm_envs

    return bool(vllm_envs.VLLM_USE_BREAKABLE_CUDAGRAPH)
