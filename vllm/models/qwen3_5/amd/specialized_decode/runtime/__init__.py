# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8 (2.4T-A95B) specialized decode on MI355X (TP8), enabled by
``--monokernel`` (``KernelConfig.monokernel``).

``specialized_decode_config`` runs in the model's ``__init__``, before the
weights load. It returns None for the stock decode, and raises with every
reason from ``unsupported`` when ``--monokernel`` is set for a deployment the
kernels cannot serve. Checks that need the loaded model run when the executor
is built (``decode.SpecializedDecode``), those that need the GDN states at
every KV cache bind (``decode.SpecializedDecode.bind_caches``)."""

from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.models.qwen3_5.amd.specialized_decode.mono import layout as L

logger = init_logger(__name__)

NAME = "Qwen3.8 specialized decode (--monokernel)"
ARCHITECTURE = "Qwen3_5MoeForCausalLM"


@dataclass(frozen=True)
class DecodeConfig:
    # rows a launch serves; a wider step runs vLLM's decode
    max_tokens: int = L.MAX_TOKENS

    @property
    def widths(self) -> tuple[int, ...]:
        """Every row count up to ``max_tokens``: eager steps and FULL graph
        capture sizes alike (a graph's pad rows are inert in the kernels)."""
        return tuple(range(1, self.max_tokens + 1))


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
    return DecodeConfig()


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

    mc, pc, cc = (
        vllm_config.model_config,
        vllm_config.parallel_config,
        vllm_config.cache_config,
    )
    oc = vllm_config.offload_config
    checks = (
        (
            mc.architecture != ARCHITECTURE,
            f"architecture {mc.architecture} ({ARCHITECTURE} only)",
        ),
        (
            pc.tensor_parallel_size != L.TP,
            f"TP {pc.tensor_parallel_size} (TP{L.TP} only)",
        ),
        (pc.pipeline_parallel_size != 1, "pipeline parallelism"),
        (pc.data_parallel_size != 1, "data parallelism"),
        (pc.enable_expert_parallel, "expert parallelism"),
        (pc.enable_eplb, "EPLB"),
        (pc.decode_context_parallel_size != 1, "decode context parallelism"),
        (pc.prefill_context_parallel_size != 1, "prefill context parallelism"),
        # the kernels read one batch's attention metadata per step
        (pc.use_ubatching, "DBO (dual-batch overlap)"),
        # the blocks take and return every token row on every rank
        (pc.use_sequence_parallel_moe, "sequence-parallel MoE"),
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
        # the GDN block runs plain decode rows, one token a request
        (vllm_config.speculative_config is not None, "speculative decoding"),
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
        # the layer block runs the shared expert as routed expert 512
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
    text = mc.hf_text_config
    shapes = text_shapes(text)
    if shapes != L.SHAPES:
        reasons.append(
            f"model shapes {shapes} ({', '.join(L.SHAPE_NAMES)}; Qwen3.8 is {L.SHAPES})"
        )
    if not getattr(text, "norm_topk_prob", True):
        reasons.append("router weights without top-k renormalization")
    if getattr(text, "layer_scale", False):
        reasons.append("layer scale")
    layers = getattr(text, "num_hidden_layers", 0)
    if layers > L.MAX_LAYERS:
        reasons.append(f"{layers} decoder layers (at most {L.MAX_LAYERS})")
    return reasons + platform_reasons()


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
