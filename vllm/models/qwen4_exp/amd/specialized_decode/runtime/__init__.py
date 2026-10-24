# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next specialized decode on MI355X (TP2), enabled by
``--monokernel`` (``KernelConfig.monokernel``).

``specialized_decode_config`` runs in the model's ``__init__``, before the
weights load. It returns None for the stock decode, and raises with every
reason from ``unsupported`` when ``--monokernel`` is set for a deployment the
kernels cannot serve. Checks that need the loaded model run when the executor
is built (``decode.SpecializedDecode``), those that need the GDN states at
every KV cache bind (``decode.SpecializedDecode.bind_caches``)."""

from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.models.qwen4_exp.amd.specialized_decode.mono import layout as L

logger = init_logger(__name__)

TP = 2
NAME = "Qwen3.8-Flash-Next specialized decode (--monokernel)"


@dataclass(frozen=True)
class DecodeConfig:
    # rows a request decodes a step: 1, or num_speculative_tokens + 1 with MTP
    q: int = 1
    # rows a launch serves; a wider step runs vLLM's decode
    max_tokens: int = L.MAX_TOKENS

    @property
    def widths(self) -> tuple[int, ...]:
        """The rows a launch serves: whole requests up to ``max_tokens`` rows,
        eager steps and FULL graph capture sizes alike (a graph's pad rows are
        inert in the kernels; with MTP a replay pads whole requests)."""
        return tuple(range(self.q, self.max_tokens + 1, self.q))


def specialized_decode_config(vllm_config) -> DecodeConfig | None:
    """The deployment's specialized decode settings, None for the stock decode.
    Raises when ``--monokernel`` is set for an unsupported deployment, listing
    every reason."""
    if not vllm_config.kernel_config.monokernel:
        return None
    reasons = unsupported(vllm_config)
    if reasons:
        raise ValueError(
            f"{NAME} does not support: " + "; ".join(reasons) + " (without "
            "--monokernel, vLLM's decode runs)"
        )
    if piecewise_graphs_only(vllm_config):
        logger.warning(
            "%s: PIECEWISE-only CUDA graphs capture every decode step as stock; "
            "use FULL decode graphs",
            NAME,
        )
    spec = vllm_config.speculative_config
    return DecodeConfig(q=1 if spec is None else spec.num_speculative_tokens + 1)


def unsupported(vllm_config) -> list[str]:
    """Every reason this deployment cannot run the specialized decode (empty: it
    can). Config only: the loaded model is checked when the executor is built.
    Default-deny: every feature below is refused, the kernels serve none of
    them."""
    from vllm import envs
    from vllm.config import CompilationMode
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,
    )
    from vllm.models.qwen4_exp.common.hyperconnection import (
        is_hc_sequence_parallel_enabled,
    )

    mc, pc, cc = (
        vllm_config.model_config,
        vllm_config.parallel_config,
        vllm_config.cache_config,
    )
    spec, oc = vllm_config.speculative_config, vllm_config.offload_config
    checks = (
        (pc.tensor_parallel_size != TP, f"TP {pc.tensor_parallel_size} (TP{TP} only)"),
        (pc.pipeline_parallel_size != 1, "pipeline parallelism"),
        (pc.data_parallel_size != 1, "data parallelism"),
        (pc.enable_expert_parallel, "expert parallelism"),
        (pc.enable_eplb, "EPLB"),
        (pc.decode_context_parallel_size != 1, "decode context parallelism"),
        (pc.prefill_context_parallel_size != 1, "prefill context parallelism"),
        # the kernels read one batch's attention metadata per step
        (pc.use_ubatching, "DBO (dual-batch overlap)"),
        # the blocks take and return every token row on every rank
        (
            is_hc_sequence_parallel_enabled(vllm_config),
            "sequence parallelism (HC or MoE)",
        ),
        (vllm_config.lora_config is not None, "LoRA"),
        (vllm_config.kv_transfer_config is not None, "KV transfer connectors"),
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
        # a verify step runs whole uniform requests of num_speculative_tokens + 1
        # rows; the drafter reads the pre-mixer state the MTP method keeps
        (
            spec is not None and spec.method != "mtp",
            f"speculative decoding method {getattr(spec, 'method', None)!r} (MTP only)",
        ),
        (
            spec is not None and spec.num_speculative_tokens + 1 > L.MAX_TOKENS,
            f"more than {L.MAX_TOKENS - 1} speculative tokens",
        ),
        (
            spec is not None and spec.enable_adaptive_verification,
            "adaptive verification",
        ),
        (
            spec is not None and spec.uses_dynamic_speculative_decoding(),
            "dynamic speculative decoding",
        ),
        # a verify step's GDN layers run vLLM's GDN core on AITER's Triton path
        (
            spec is not None and not gdn_aiter_triton_available(),
            "MTP without AITER's Triton GDN kernels",
        ),
        # the GDN block reads and writes one state slot a request in place
        (
            cc.mamba_cache_mode != "none",
            (
                f"GDN prefix caching (mamba_cache_mode {cc.mamba_cache_mode!r}; "
                "pass --no-enable-prefix-caching)"
            ),
        ),
        (
            MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                mc.dtype, cc.mamba_cache_dtype, cc.mamba_ssm_cache_dtype
            )
            != (L.CONV_DTYPE, L.STATE_DTYPE),
            (
                "GDN cache dtypes other than a BF16 conv state and an FP32 "
                "recurrent state (mamba cache dtype)"
            ),
        ),
        # the MoE block runs the shared expert as routed expert 512
        (
            not envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS,
            (
                "an unfused shared expert (set "
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1)"
            ),
        ),
        # the model state builds the executor and binds it to each KV cache
        (
            not vllm_config.use_v2_model_runner,
            "the V1 model runner (set VLLM_USE_V2_MODEL_RUNNER=1)",
        ),
        # plan() and the launches run in forward; a compiled forward would
        # trace them
        (
            vllm_config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE,
            "torch.compile (compilation mode STOCK_TORCH_COMPILE)",
        ),
    )
    reasons = [why for bad, why in checks if bad]
    shapes = text_shapes(mc.hf_text_config)
    if shapes != L.SHAPES:
        reasons.append(
            f"model shapes {shapes} ({', '.join(L.SHAPE_NAMES)}; "
            f"Qwen3.8-Flash-Next is {L.SHAPES})"
        )
    if not getattr(mc.hf_text_config, "norm_topk_prob", True):
        reasons.append("router weights without top-k renormalization")
    layers = getattr(mc.hf_text_config, "num_hidden_layers", 0)
    if layers > L.MAX_LAYERS:
        reasons.append(f"{layers} decoder layers (at most {L.MAX_LAYERS})")
    return reasons + platform_reasons()


def gdn_aiter_triton_available() -> bool:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        GDN_AITER_TRITON_AVAILABLE,
    )

    return GDN_AITER_TRITON_AVAILABLE


def text_shapes(text) -> tuple:
    return tuple(getattr(text, name, None) for name in L.SHAPE_NAMES)


def platform_reasons() -> list[str]:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return ["this platform (ROCm only)"]
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return ["this GPU (gfx950 / MI355X only)"]
    import torch

    cus = current_platform.num_compute_units(torch.accelerator.current_device_index())
    if cus != L.BLOCKS:
        # every CTA of a launch must be resident: a partitioned GPU deadlocks
        return [f"{cus} CUs (a partitioned GPU; the kernels need {L.BLOCKS})"]
    return []


def piecewise_graphs_only(vllm_config) -> bool:
    """Decode steps captured as PIECEWISE graphs only: plan() runs them stock."""
    from vllm.config import CUDAGraphMode

    mode = vllm_config.compilation_config.cudagraph_mode
    if mode is None or mode == CUDAGraphMode.NONE:
        return False
    return mode.decode_mode() != CUDAGraphMode.FULL
