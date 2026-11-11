# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.2 specialized decode: one FlyDSL MonoKernel per MoE decoder layer inside the
stock layer loop (layer mode), enabled by
`additional_config={"decode_backend": "latency_specialized"}`. Options (the
`DecodeConfig` fields) go under `additional_config["specialized_decode"]`.

`specialized_decode_config` runs in the model's `__init__`, before the weights
load. It returns None for the stock decode, and raises with every reason from
`unsupported` when the specialized decode is requested for a deployment it cannot
serve. Checks that need the loaded weights run when the executor is built
(`decode.SpecializedDecode`), those that need the KV cache in `prepare_kv`."""

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
    """Step and kernel options. `decode_config` fills it from the defaults, the
    `additional_config["specialized_decode"]` options and vLLM's max_model_len."""

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
    """The requested decode backend: the one place that reads the switch.
    `--monokernel` (`KernelConfig.monokernel`) asks for the specialized decode as
    `additional_config={"decode_backend": "latency_specialized"}` does."""
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", AUTO) if isinstance(extra, dict) else AUTO
    kernel_config = getattr(vllm_config, "kernel_config", None)
    if backend == AUTO and getattr(kernel_config, "monokernel", False):
        return SPECIALIZED
    return backend


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
    """The defaults, overridden by `additional_config["specialized_decode"]`. The
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
    """`unsupported`'s reasons and the `DecodeConfig` (None if the options are
    invalid), so that `decode_config` runs once."""
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
            kv_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"),
            f"kv_cache_dtype {kv_dtype} (bfloat16 or fp8 e4m3)",
        ),
        # an fp8 cache is written and read with scale 1 (checked again at bind)
        (
            kv_dtype.startswith("fp8")
            and bool(getattr(vllm_config.cache_config, "calculate_kv_scales", False)),
            "calculate_kv_scales (the kernels use KV scale 1)",
        ),
        # plan() and the per-layer kernels run in forward; a compiled forward would
        # trace them (DeepseekV32Model has no @support_torch_compile)
        (
            vllm_config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE,
            "torch.compile (compilation mode STOCK_TORCH_COMPILE)",
        ),
    )
    reasons = [why for bad, why in checks if bad]
    if model_type == "glm_moe_dsa":
        reasons += geometry_reasons(mc.hf_config)
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


# The checkpoint geometry the kernels are compiled for (flydsl/config.py GLM5_CONFIG,
# the 2048-key top-k, the 78-layer / 3-dense layer loop). glm_moe_dsa alone does not
# pin these: another DSA checkpoint of the same model_type would start and then read
# its weights with the wrong strides.
GEOMETRY = {
    "hidden_size": 6144,
    "num_hidden_layers": 78,
    "first_k_dense_replace": 3,
    "num_attention_heads": 64,
    "q_lora_rank": 2048,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 192,
    "v_head_dim": 256,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "moe_intermediate_size": 2048,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "index_topk": 2048,
    "index_n_heads": 32,
    "index_head_dim": 128,
}


def geometry_reasons(hf_config) -> list[str]:
    """Every config value that differs from the geometry the kernels are built for."""
    return [
        f"{key} {getattr(hf_config, key, None)!r} (the kernels need {want!r})"
        for key, want in GEOMETRY.items()
        if getattr(hf_config, key, None) != want
    ]


def builds_indexer(hf_config, layer: int) -> bool:
    """Whether decoder layer `layer` has its own indexer, by the IndexCache rule of
    `DeepseekV2MLAAttention` (`index_topk_freq`, `index_skip_topk_offset`,
    `index_topk_pattern`): the other layers reuse the previous indexer's top-k and have
    no `indexer.*` linears, so their (absent) `wq_b` has no quantization to check."""
    pattern = getattr(hf_config, "index_topk_pattern", None)
    if pattern is not None:
        return not (0 <= layer < len(pattern) and pattern[layer] == "S")
    freq = getattr(hf_config, "index_topk_freq", 1)
    offset = getattr(hf_config, "index_skip_topk_offset", 2)
    return max(layer - offset + 1, 0) % freq == 0


def attention_quant_reasons(vllm_config, layers) -> list[str]:
    """Why the mono layers' attention linears cannot be read as checkpointed: each must
    be BF16 (excluded from quantization), Quark per-channel FP8 with per-token dynamic
    activations (PTPC), which the kernel reads with its per-row scales, or Quark 128 x
    128 block FP8 with per-128 dynamic activations (GLM-5.3), read with its block
    scales; the attention proper (all but `indexer.wq_b`) one format on every mono
    layer, and an FP8 `wq_b` the attention's FP8 format. Per-tensor FP8 and non-Quark
    quantization (e.g. zai-org/GLM-5.2-FP8) are refused: the specialized decode never
    requantizes weights."""
    qc = getattr(vllm_config, "quant_config", None)
    if qc is None:
        return []
    if qc.get_name() != "quark":
        name = qc.get_name()
        return [f"quantization {name} (Quark with BF16 or per-channel FP8 attention)"]
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
        kFp8DynamicTokenSym,
        kFp8Static128BlockE8M0Sym,
        kFp8Static128BlockSym,
        kFp8StaticChannelSym,
    )

    formats = {
        (kFp8StaticChannelSym, kFp8DynamicTokenSym): "per-channel FP8",
        (kFp8Static128BlockSym, kFp8Dynamic128Sym): "block FP8",
        (kFp8Static128BlockE8M0Sym, kFp8Dynamic128Sym): "block FP8",
    }
    bad: dict[str, set[int]] = {}
    # (layer, linear) -> format of the attention proper's linears
    fmt: dict[int, dict[str, str]] = {}
    index_fmt: dict[int, str] = {}  # FP8 indexer wq_b
    hf = vllm_config.model_config.hf_config
    for L in layers:
        for proj in ATTENTION_LINEARS:
            if proj == "indexer.wq_b" and not builds_indexer(hf, L):
                continue
            prefix = f"model.layers.{L}.self_attn.{proj}"
            try:
                wk, ak, method = qc.get_quant_method_target(prefix, LinearBase)
            except (NotImplementedError, ValueError) as e:
                why = f"{proj}: {e}"
            else:
                f = (
                    "BF16"
                    if method is UnquantizedLinearMethod
                    else formats.get((wk, ak))
                )
                if f is not None:
                    if proj != "indexer.wq_b":
                        fmt.setdefault(L, {})[proj] = f
                    elif f != "BF16":
                        index_fmt[L] = f
                    continue
                why = f"{proj} quantized as {wk} / {ak}"
            bad.setdefault(why, set()).add(L)
    reasons = [
        f"{why} on layers {sorted(Ls)[0]}..{sorted(Ls)[-1]} (attention must be BF16, "
        "per-channel FP8 with per-token activations or 128 x 128 block FP8 with "
        "per-128 activations)"
        for why, Ls in bad.items()
    ]
    partial = sorted(L for L, d in fmt.items() if len(set(d.values())) > 1)
    attn_fmt = {L: next(iter(d.values())) for L, d in fmt.items() if d}
    if partial:
        d = fmt[partial[0]]
        reasons.append(
            f"partial FP8 attention on layers {partial[0]}..{partial[-1]} (layer "
            f"{partial[0]}: "
            + ", ".join(
                f"{f} {[p for p, v in d.items() if v == f]}"
                for f in sorted(set(d.values()))
            )
            + "; a layer's attention is read all BF16 or all one FP8 format)"
        )
    elif len(set(attn_fmt.values())) > 1:
        by: dict[str, list[int]] = {}
        for L, f in sorted(attn_fmt.items()):
            by.setdefault(f, []).append(L)
        reasons.append(
            "mixed attention precision across mono layers ("
            + "; ".join(
                f"{f} on {len(Ls)} layers, {Ls[0]}..{Ls[-1]}" for f, Ls in by.items()
            )
            + ")"
        )
    odd = sorted(
        L for L, f in index_fmt.items() if attn_fmt.get(L, "BF16") not in ("BF16", f)
    )
    if odd:
        reasons.append(
            f"indexer.wq_b {index_fmt[odd[0]]} on layers {odd[0]}..{odd[-1]} with "
            f"{attn_fmt[odd[0]]} attention (the kernel reads one FP8 scale format)"
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
