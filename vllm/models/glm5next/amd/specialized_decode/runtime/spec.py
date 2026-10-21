# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Side-effect-free configuration of the GLM-5.3-Flash specialized decode.

``specialized_decode_config`` runs in the model's ``__init__``, before the weights
load. It returns None unless ``--monokernel`` is set, and raises with every reason
when the deployment cannot run it. Checks that need the loaded weights run when the
executor is built (``guards.model_refusals``), those that need the KV caches when
they are bound (``decode.SpecializedDecode.bind_caches``)."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.models.glm5next.amd.specialized_decode.mono.config import KPOOL
from vllm.models.glm5next.amd.specialized_decode.mono.ffn.layout import POLL_STAGES

if TYPE_CHECKING:
    from vllm.config import VllmConfig

TP_SIZES = (2, 4, 8)
WIDTHS = (2, 4, 8)
# wider steps at these TP sizes are slower than vLLM's path
TP_MAX_WIDTH = {2: 4}
# step token counts the kernels take (a step pads up to the next width)
STEP_TOKENS = (1, 2, 4, 8)
# whole-layer launches take at most this many rows
MAX_ROWS = 8
# one persistent launch: 256 CTAs, every one resident
GRID = 256
NUMERICS = (
    "expert inputs quantized to MXFP8 (vLLM: MXFP4)",
    "attention TP reduce summed in BF16 inside the launch",
    "KDA conv taps, L2 norms, gates and recurrence in FP32, as vLLM's kernels",
    "sparse-attention FP8 projections on bf16 activations (vLLM: FP8 per 128)",
    "W_UK / W_UV in bf16 from kv_b_proj (vLLM: AITER FP4 / FP8 BMM)",
    "split-attention partial sums stored in bf16 before the merge",
    (
        "a 2048-token selection a tail overflows drops the last positions of its "
        "lowest-scored pool (vLLM: the last of its top-k order)"
    ),
    (
        "an e4m3 KV cache is read back to bf16 for the attention (vLLM: fp8 query "
        "and fp8 MFMA)"
    ),
)
# bf16, or e4m3 over a per-tensor scale as vLLM's ROCm sparse MLA stores it
KV_DTYPES = ("auto", "bfloat16", "fp8", "fp8_e4m3")
GEOMETRY = {
    "hidden_size": 4096,
    "n_routed_experts": 288,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "moe_intermediate_size": 2048,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "swiglu_limit": 10.0,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "mhc": True,
    "linear_num_heads": 64,
    "linear_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "linear_lower_bound": -5.0,
}


class Feature(enum.Enum):
    """A deployment feature the specialized decode does not support (refused when
    in use)."""

    PIPELINE_PARALLEL = "pipeline parallel"
    DATA_PARALLEL = "data parallel"
    EXPERT_PARALLEL = "expert parallel"
    EPLB = "EPLB"
    DECODE_CONTEXT_PARALLEL = "decode context parallel"
    PREFILL_CONTEXT_PARALLEL = "prefill context parallel"
    LORA = "LoRA"
    KV_CONNECTOR = "KV transfer connector"
    ROUTED_EXPERT_CAPTURE = "routed-expert capture"
    SPECULATIVE_DECODING = "speculative decoding other than MTP"
    SLEEP_MODE = "sleep mode"
    CPU_WEIGHT_OFFLOAD = "CPU weight offload"
    WEIGHT_UPDATES = "weight updates"
    PROMPT_EMBEDS = "prompt embeds"


@dataclass(frozen=True)
class DecodeConfig:
    """The deployment's specialized decode: built launch widths (rows), the step
    token counts that run the kernels, and the rows a request owns (1, or with MTP
    a verify step's token and its drafts)."""

    tp_size: int
    widths: tuple[int, ...] = WIDTHS
    tokens: tuple[int, ...] = STEP_TOKENS
    tpr: int = 1
    # a kernel wait expires (and the engine fail-stops) only when a peer is this
    # late: a rank seconds behind (a host stall) is waited out, as vLLM's own
    # collectives do, and a stalled peer still stops the engine inside vLLM's
    # execute timeout
    poll_timeout_us: int = 60_000_000
    numerics: tuple[str, ...] = NUMERICS

    def width_for(self, tokens: int) -> int | None:
        return next((w for w in sorted(self.widths) if w >= tokens), None)


def mtp_tokens(vllm_config: VllmConfig) -> int:
    """Rows a request in a verify step: its token and its MTP drafts (1 without)."""
    sc = vllm_config.speculative_config
    if sc is None or getattr(sc, "method", None) != "mtp":
        return 1
    return 1 + sc.num_speculative_tokens


def tail_ring(tpr: int) -> int:
    """The kpool tail ring vLLM sizes for ``tpr - 1`` drafts (``Glm5NextTailCache``)."""
    span = -(-(KPOOL + tpr - 1) // KPOOL)
    return KPOOL * (1 << (span - 1).bit_length())


def step_tokens(vllm_config: VllmConfig) -> tuple[int, ...]:
    """The step token counts the kernels take: decode steps, or with MTP whole
    verify steps of at most the TP size's row cap."""
    t = mtp_tokens(vllm_config)
    if t == 1:
        return STEP_TOKENS
    tp = vllm_config.parallel_config.tensor_parallel_size
    cap = min(TP_MAX_WIDTH.get(tp, MAX_ROWS), MAX_ROWS)
    return tuple(r * t for r in STEP_TOKENS if r * t <= cap)


def speculative_reasons(vllm_config: VllmConfig) -> list[str]:
    """MTP runs as uniform verify steps; anything else speculative is refused."""
    sc = vllm_config.speculative_config
    if sc is None or getattr(sc, "method", None) != "mtp":
        return []
    reasons = []
    if not step_tokens(vllm_config):
        reasons.append(
            f"MTP with {sc.num_speculative_tokens} drafts: one request's verify step "
            "is wider than the kernel's rows at this TP size"
        )
    if getattr(sc, "enable_adaptive_verification", False):
        reasons.append("adaptive verification (non-uniform verify steps)")
    dynamic = getattr(sc, "uses_dynamic_speculative_decoding", None)
    if callable(dynamic) and dynamic():
        reasons.append("dynamic speculative decoding")
    if getattr(vllm_config.cache_config, "use_kda_recoverssm", False):
        reasons.append("KDA RecoverSSM (needs the per-draft state slots)")
    return reasons


def kernel_layers(hf: Any) -> list[int]:
    """The MoE layers, except the last (its mHC post contracts the streams)."""
    types = getattr(hf, "mlp_layer_types", None) or []
    last = hf.num_hidden_layers - 1
    return [i for i, t in enumerate(types[:last]) if t == "sparse"]


def is_kda(layer: Any) -> bool:
    return getattr(layer, "layer_kind", None) == "kda"


def is_dsa(layer: Any) -> bool:
    return getattr(layer, "layer_kind", None) == "mla"


def features_in_use(vllm_config: VllmConfig) -> list[Feature]:
    pc = vllm_config.parallel_config
    mc = vllm_config.model_config
    offload = vllm_config.offload_config
    used = {
        Feature.PIPELINE_PARALLEL: pc.pipeline_parallel_size > 1,
        Feature.DATA_PARALLEL: pc.data_parallel_size > 1,
        Feature.EXPERT_PARALLEL: pc.enable_expert_parallel,
        Feature.EPLB: pc.enable_eplb,
        Feature.DECODE_CONTEXT_PARALLEL: pc.decode_context_parallel_size > 1,
        Feature.PREFILL_CONTEXT_PARALLEL: pc.prefill_context_parallel_size > 1,
        Feature.LORA: vllm_config.lora_config is not None,
        Feature.KV_CONNECTOR: vllm_config.kv_transfer_config is not None,
        Feature.ROUTED_EXPERT_CAPTURE: (
            vllm_config.aux_output_config.enable_return_routed_experts
        ),
        Feature.SPECULATIVE_DECODING: (
            vllm_config.speculative_config is not None
            and getattr(vllm_config.speculative_config, "method", None) != "mtp"
        ),
        Feature.SLEEP_MODE: mc.enable_sleep_mode,
        Feature.CPU_WEIGHT_OFFLOAD: (
            offload.uva.cpu_offload_gb > 0 or offload.prefetch.offload_group_size > 0
        ),
        Feature.WEIGHT_UPDATES: vllm_config.weight_transfer_config is not None,
        Feature.PROMPT_EMBEDS: mc.enable_prompt_embeds,
    }
    return [f for f, on in used.items() if on]


def graph_reasons(
    vllm_config: VllmConfig, tokens: tuple[int, ...], tpr: int = 1
) -> list[str]:
    """Graph modes and capture sizes the decision cannot be baked into (verify
    steps: vLLM rounds decode graphs up to whole requests itself)."""
    from vllm import envs
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    cc = vllm_config.compilation_config
    reasons = []
    if cc.mode == CompilationMode.STOCK_TORCH_COMPILE:
        reasons.append("stock torch.compile")
    mode = cc.cudagraph_mode
    if mode is not None and mode != CUDAGraphMode.NONE:
        if mode.decode_mode() == CUDAGraphMode.PIECEWISE:
            reasons.append(
                "piecewise-only CUDA graphs carry no attention metadata "
                "(use FULL_AND_PIECEWISE, FULL_DECODE_ONLY or FULL)"
            )
        bad = [
            s
            for s in cc.cudagraph_capture_sizes or ()
            if tpr == 1 and s <= max(tokens) and s not in tokens
        ]
        if bad:
            reasons.append(
                f"CUDA graph capture sizes {bad} are not kernel step sizes "
                f"{sorted(tokens)}"
            )
    if envs.VLLM_USE_BREAKABLE_CUDAGRAPH:
        reasons.append("breakable CUDA graphs")
    return reasons


def refusal_reasons(vllm_config: VllmConfig) -> list[str]:
    """Every reason this deployment cannot run the specialized decode (empty: it
    can). Config only: the loaded weights are checked when the executor is built."""
    from vllm.platforms import current_platform

    hf = vllm_config.model_config.hf_text_config
    reasons = []
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            reasons.append("needs gfx950")
    else:
        reasons.append("needs ROCm")
    if not vllm_config.use_v2_model_runner:
        reasons.append("needs Model Runner V2")
    reasons += [f.value for f in features_in_use(vllm_config)]
    reasons += speculative_reasons(vllm_config)
    tokens = step_tokens(vllm_config)
    if tokens:
        reasons += graph_reasons(vllm_config, tokens, mtp_tokens(vllm_config))
    if getattr(hf, "model_type", None) != "glm5_next_text":
        reasons.append("needs glm5_next")
    else:
        for key, want in GEOMETRY.items():
            if getattr(hf, key, None) != want:
                reasons.append(f"needs {key} {want}, got {getattr(hf, key, None)}")
    if vllm_config.parallel_config.tensor_parallel_size not in TP_SIZES:
        reasons.append("needs tensor parallel size 2, 4 or 8")
    if vllm_config.parallel_config.use_sequence_parallel_moe:
        reasons.append("sequence-parallel MoE")
    if vllm_config.cache_config.cache_dtype not in KV_DTYPES:
        reasons.append(
            f"{vllm_config.cache_config.cache_dtype} KV cache (needs bf16 or e4m3)"
        )
    ssm = getattr(vllm_config.cache_config, "mamba_ssm_cache_dtype", "auto")
    if ssm not in ("auto", "float32"):
        reasons.append(f"{ssm} KDA recurrent state (needs float32)")
    return reasons


def specialized_decode_config(vllm_config: VllmConfig) -> DecodeConfig | None:
    """The deployment's specialized decode, None without ``--monokernel``. Raises
    when it is requested for a deployment it cannot serve, listing every reason."""
    if not vllm_config.kernel_config.monokernel:
        return None
    reasons = refusal_reasons(vllm_config)
    if reasons:
        raise ValueError(
            "--monokernel cannot run the GLM-5.3-Flash specialized decode: "
            + "; ".join(reasons)
            + "."
        )
    tp = vllm_config.parallel_config.tensor_parallel_size
    tpr = mtp_tokens(vllm_config)
    if tpr == 1:
        return DecodeConfig(tp_size=tp)
    tokens = step_tokens(vllm_config)
    return DecodeConfig(tp_size=tp, widths=tokens, tokens=tokens, tpr=tpr)


# the kernels' poll stages, in the order of the error words
STAGES = POLL_STAGES
