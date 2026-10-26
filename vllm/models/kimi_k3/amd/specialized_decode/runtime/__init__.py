# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 specialized decode on MI355X (layer mode), enabled by
``additional_config={"decode_backend": "latency_specialized"}``.

``specialized_decode_config`` runs in the model's ``__init__``, before the weights
load. It returns None for the stock decode, and raises with every reason from
``unsupported`` when the specialized decode is requested for a deployment it cannot
serve. Checks that need the loaded weights run when the executor is built
(``decode.SpecializedDecode``), those that need the KV cache in ``prepare_kv``."""

from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

AUTO, SPECIALIZED = "auto", "latency_specialized"
TP = 8
DCP = TP  # the MLA layers' KV shards: one a TP rank
DCP_SIZES = (1, DCP)  # M1's builds: the whole KV a rank, or DCP over the TP group
# the kernels' rows: the MoE route table's token bitmaps (scratch and peer memory
# are sized for it)
MAX_TOKENS = 64
# a served step's rows: at 32 and 64 rows the kernels lose to vLLM's decode
MAX_ROWS = 16
# the served max_num_seqs: a full step of each is one of the widths (MAX_ROWS)
MAX_SEQS = (1, 2, 4, 8)
# M1c's qlens: DCP1 runs only them (AITER's asm decode at DCP1 is not served)
M1C_QLENS = (4, 8)
WHOLE, FFN = "whole", "ffn"
# Kimi-K3's text shapes the kernels are written for (hidden size, routed experts,
# KDA heads, KDA head dim, KDA short-conv width)
K3_SHAPES = (7168, 896, 96, 128, 4)


@dataclass(frozen=True)
class DecodeConfig:
    # rows a request verifies: num_speculative_tokens + 1
    q: int
    # the MLA layers run M1 (else vLLM's attention ops, then K2; ``m1_reasons``)
    m1: bool = True
    # the DCP KV cache interleave (AITER's decode path, M1a's query pad)
    interleave: int = 1

    @property
    def widths(self) -> tuple[int, ...]:
        """The rows a launch serves: whole requests, at most MAX_ROWS rows. Every
        uniform verify step up to MAX_ROWS rows is one of them, so an eager step
        never pads; a FULL replay pads whole requests the way vLLM's stock decode
        does."""
        return tuple(range(self.q, MAX_ROWS + 1, self.q))


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
    reasons = unsupported(vllm_config)
    if reasons:
        raise ValueError(
            "Kimi-K3 specialized decode (decode_backend=latency_specialized) does "
            "not support: " + "; ".join(reasons) + f" (decode_backend={AUTO} runs "
            "vLLM's decode)"
        )
    if piecewise_graphs_only(vllm_config):
        logger.warning(
            "Kimi-K3 specialized decode: PIECEWISE-only CUDA graphs capture every "
            "decode step as stock; use FULL decode graphs"
        )
    m1 = m1_reasons(vllm_config)
    if m1:
        logger.info(
            "Kimi-K3 specialized decode: the MLA layers run vLLM's attention ops, "
            "then K2 (M1 needs %s)",
            "; ".join(m1),
        )
    spec = vllm_config.speculative_config
    return DecodeConfig(
        q=spec.num_speculative_tokens + 1,
        m1=not m1,
        interleave=vllm_config.parallel_config.cp_kv_cache_interleave_size,
    )


def m1_reasons(vllm_config) -> list[str]:
    """Why the MLA layers cannot run M1 (empty: they can). At DCP8 not a refusal:
    those layers run vLLM's attention ops instead, then K2 (``step_reasons``)."""
    import vllm.envs as envs

    cc = vllm_config.cache_config
    checks = (
        # M1a writes the MLA layers' fp8 KV and hands AITER an fp8 query
        (not cc.cache_dtype.startswith("fp8"), "an fp8 KV cache"),
        # M1 runs W_UK / W_UV as AITER's MXFP4 bmms
        (
            not envs.VLLM_ROCM_USE_AITER_FP4BMM,
            "AITER's MXFP4 bmms (VLLM_ROCM_USE_AITER_FP4BMM=1)",
        ),
    )
    return [why for bad, why in checks if bad]


def unsupported(vllm_config) -> list[str]:
    """Every reason this deployment cannot run the specialized decode (empty: it
    can). Config only: the loaded weights are checked when the executor is built.
    Default-deny: every feature below is refused, the kernels serve none of them.
    What only M1 needs is not refused at DCP8 (``m1_reasons``)."""
    from vllm.config import CompilationMode
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,
    )

    mc, pc, cc = (
        vllm_config.model_config,
        vllm_config.parallel_config,
        vllm_config.cache_config,
    )
    spec, oc = vllm_config.speculative_config, vllm_config.offload_config
    kv = vllm_config.kv_transfer_config
    checks = (
        (pc.tensor_parallel_size != TP, f"TP {pc.tensor_parallel_size} (TP{TP} only)"),
        (pc.pipeline_parallel_size != 1, "pipeline parallelism"),
        (pc.data_parallel_size != 1, "data parallelism"),
        (pc.enable_expert_parallel, "expert parallelism"),
        (pc.enable_eplb, "EPLB"),
        (
            pc.decode_context_parallel_size not in DCP_SIZES,
            f"decode context parallelism other than 1 or {DCP} (over the TP group)",
        ),
        (pc.prefill_context_parallel_size != 1, "prefill context parallelism"),
        # the kernels read one batch's attention metadata per step
        (pc.use_ubatching, "DBO (dual-batch overlap)"),
        (vllm_config.lora_config is not None, "LoRA"),
        # MoRIIO consumers finish loads before execution, without per-layer hooks.
        (
            kv is not None
            and (
                kv.kv_connector != "MoRIIOConnector"
                or kv.kv_connector_module_path is not None
                or kv.kv_role != "kv_consumer"
            ),
            "KV transfer connectors other than a MoRIIO consumer",
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
        # the kernels serve spec-verify steps of whole uniform requests only
        (spec is None, "decoding without speculative decoding"),
        (
            spec is not None and spec.num_speculative_tokens + 1 > MAX_ROWS,
            f"more than {MAX_ROWS - 1} speculative tokens",
        ),
        (
            spec is not None and spec.enable_adaptive_verification,
            "adaptive verification",
        ),
        (
            spec is not None and spec.uses_dynamic_speculative_decoding(),
            "dynamic speculative decoding",
        ),
        # K1 keeps a BF16 conv state and an FP32 recurrent state in place; it
        # maintains no ReplaySSM / RecoverSSM records
        (
            MambaStateDtypeCalculator.kda_state_dtype(mc.dtype, cc.mamba_cache_dtype)
            != (torch.bfloat16, torch.float32),
            "KDA cache dtypes other than a BF16 conv state and an FP32 recurrent "
            "state (mamba cache dtype)",
        ),
        (cc.use_replayssm or cc.use_kda_recoverssm, "ReplaySSM / RecoverSSM"),
        # the model state hooks build the executor and bind it to each KV cache
        (
            not vllm_config.use_v2_model_runner,
            "the V1 model runner (set VLLM_USE_V2_MODEL_RUNNER=1)",
        ),
        # plan() and the layer kernels run in forward; a compiled forward would
        # trace them
        (
            vllm_config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE,
            "torch.compile (compilation mode STOCK_TORCH_COMPILE)",
        ),
    )
    reasons = [why for bad, why in checks if bad]
    shapes = text_shapes(mc.hf_text_config)
    if shapes != K3_SHAPES:
        reasons.append(
            f"model shapes {shapes} (hidden, experts, KDA heads, KDA head dim, conv "
            f"width; Kimi-K3 is {K3_SHAPES})"
        )
    if getattr(mc.hf_text_config, "attn_res_block_size", None) is None:
        reasons.append("models without AttnRes")
    platform = platform_reasons()
    if spec is not None and spec.num_speculative_tokens + 1 <= MAX_ROWS:
        reasons += step_reasons(vllm_config, check_m1c=not platform)
    return reasons + platform


def m1c_serves(q: int, seqs: int) -> bool:
    """M1c runs every step of 1 .. ``seqs`` requests at qlen ``q``: its KV splits
    share out evenly over the requests (``layers.m1c_at``)."""
    if q not in M1C_QLENS:
        return False
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import mla_attn

    return all(mla_attn.splits(q) % n == 0 for n in range(1, seqs + 1))


def step_reasons(vllm_config, check_m1c: bool = True) -> list[str]:
    """The (max_num_seqs, speculative tokens) pairs served: a full step fits one
    launch, and at DCP1 every step runs M1c (``check_m1c``: M1c's geometry needs
    the kernels' modules, so only on a supported platform)."""
    k = vllm_config.speculative_config.num_speculative_tokens
    seqs = vllm_config.scheduler_config.max_num_seqs
    if seqs not in MAX_SEQS or seqs * (k + 1) > MAX_ROWS:
        return [
            f"max_num_seqs {seqs} at {k} speculative tokens (max_num_seqs 1, 2, 4 "
            f"or 8 with max_num_seqs x (k + 1) <= {MAX_ROWS})"
        ]
    if vllm_config.parallel_config.decode_context_parallel_size != 1:
        return []
    m1 = m1_reasons(vllm_config)
    if m1:
        return ["DCP1 without M1 (" + "; ".join(m1) + ")"]
    if check_m1c and not m1c_serves(k + 1, seqs):
        return [
            f"DCP1 at {k} speculative tokens, max_num_seqs {seqs} (DCP1 runs M1c "
            "only: 3 or 7 speculative tokens, max_num_seqs 1 or 2; use DCP8)"
        ]
    return []


def text_shapes(text) -> tuple:
    kda = getattr(text, "linear_attn_config", None) or {}
    return (
        getattr(text, "hidden_size", None),
        getattr(text, "num_experts", None),
        kda.get("num_heads"),
        kda.get("head_dim"),
        kda.get("short_conv_kernel_size"),
    )


def platform_reasons() -> list[str]:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return ["this platform (ROCm only)"]
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return ["this GPU (gfx950 / MI355X only)"]
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import BLOCKS

    cus = torch.cuda.get_device_properties().multi_processor_count
    if cus != BLOCKS:
        # every CTA of a launch must be resident: a partitioned GPU deadlocks
        return [f"{cus} CUs (a partitioned GPU; the kernels need {BLOCKS})"]
    return []


def piecewise_graphs_only(vllm_config) -> bool:
    """Decode steps captured as PIECEWISE graphs only: plan() runs them stock."""
    from vllm.config import CUDAGraphMode

    mode = vllm_config.compilation_config.cudagraph_mode
    if mode is None or mode == CUDAGraphMode.NONE:
        return False
    return mode.decode_mode() != CUDAGraphMode.FULL


def layer_kinds(config) -> dict[int, str]:
    """Decoder layer -> kind, for every layer a kernel serves: the MoE layers with
    AttnRes and at least one block before them, minus the MLA block-write layers
    (none in Kimi-K3)."""
    block = config.attn_res_block_size
    if block is None or config.num_experts is None:
        return {}
    out = {}
    for i in range(1, config.num_hidden_layers):
        if i < config.first_k_dense_replace or i % config.moe_layer_freq:
            continue
        if config.is_kda_layer(i):
            out[i] = WHOLE
        elif i % block:
            out[i] = FFN
    return out
