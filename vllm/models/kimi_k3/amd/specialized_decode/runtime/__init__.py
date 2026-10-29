# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 specialized decode on MI355X and MI325X (layer mode), enabled by
``additional_config={"decode_backend": "latency_specialized"}``.

``specialized_decode_config`` runs in the model's ``__init__``, before the weights
load. It returns None for the stock decode, and raises with every reason from
``unsupported`` when the specialized decode is requested for a deployment it cannot
serve. Checks that need the loaded weights run when the executor is built
(``decode.SpecializedDecode``), those that need the KV cache in ``prepare_kv``."""

import os
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

AUTO, SPECIALIZED = "auto", "latency_specialized"
TP = 8
# a launch's x rows fit one CTA's LDS
MAX_TOKENS = 8
WHOLE, FFN = "whole", "ffn"
# Kimi-K3's text shapes the kernels are written for (hidden size, routed experts,
# KDA heads, KDA head dim, KDA short-conv width)
K3_SHAPES = (7168, 896, 96, 128, 4)


@dataclass(frozen=True)
class DecodeConfig:
    # rows a request verifies: num_speculative_tokens + 1
    q: int

    @property
    def widths(self) -> tuple[int, ...]:
        """The rows a launch serves: whole requests, at most MAX_TOKENS rows. Every
        uniform verify step up to MAX_TOKENS rows is one of them, so an eager step
        never pads; a FULL replay pads whole requests the way vLLM's stock decode
        does."""
        return tuple(range(self.q, MAX_TOKENS + 1, self.q))


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
    spec = vllm_config.speculative_config
    return DecodeConfig(q=spec.num_speculative_tokens + 1)


def unsupported(vllm_config) -> list[str]:
    """Every reason this deployment cannot run the specialized decode (empty: it
    can). Config only: the loaded weights are checked when the executor is built.
    Default-deny: every feature below is refused, the kernels serve none of them."""
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
            pc.decode_context_parallel_size not in (1, pc.tensor_parallel_size),
            "decode context parallelism other than 1 or TP",
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
            spec is not None and spec.num_speculative_tokens + 1 > MAX_TOKENS,
            f"more than {MAX_TOKENS - 1} speculative tokens",
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
    return reasons + platform_reasons(vllm_config)


def text_shapes(text) -> tuple:
    kda = getattr(text, "linear_attn_config", None) or {}
    return (
        getattr(text, "hidden_size", None),
        getattr(text, "num_experts", None),
        kda.get("num_heads"),
        kda.get("head_dim"),
        kda.get("short_conv_kernel_size"),
    )


def platform_reasons(vllm_config) -> list[str]:
    """The GPU's reasons: gfx950, or gfx942 with the routed experts requantized
    to packed int4 (it has no MXFP4 MFMA), a CU for every CTA, and no ``ARCH``
    that makes FlyDSL build the kernels for another GPU."""
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return ["this platform (ROCm only)"]
    from vllm.platforms.rocm import on_gfx942, on_gfx950

    if not (on_gfx950() or on_gfx942()):
        return ["this GPU (gfx950 / MI355X and gfx942 / MI325X only)"]
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import BLOCKS

    reasons = []
    props = torch.cuda.get_device_properties()
    cus = props.multi_processor_count
    if cus < BLOCKS:
        # every CTA of a launch must be resident: a partitioned GPU deadlocks
        reasons.append(f"{cus} CUs (a partitioned GPU; the kernels need {BLOCKS})")
    # FlyDSL builds for $ARCH (``mono.common.arch``); these checks and vLLM's MoE
    # follow the GPU
    arch, gpu = os.environ.get("ARCH"), props.gcnArchName.split(":")[0].lower()
    if arch and arch.lower().split(":")[0] != gpu:
        reasons.append(
            f"ARCH={arch} (FlyDSL would build the kernels for it, not {gpu})"
        )
    if on_gfx942() and not int4_experts(vllm_config):
        reasons.append(
            "gfx942 without packed-int4 routed experts (--quantization-config "
            '\'{"moe": {"weight": "int4_per_group_32"}}\')'
        )
    return reasons


def int4_experts(vllm_config) -> bool:
    """The routed experts are requantized to packed int4 (gfx942's MoE kernels)."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt4Static32,
    )

    quant = getattr(vllm_config.model_config, "quantization_config", None)
    return getattr(getattr(quant, "moe", None), "weight", None) == kInt4Static32


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
