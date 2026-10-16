# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3's specialized decode."""

from typing import TYPE_CHECKING

import torch

from vllm.models.common.specialized_decode.spec import DecodeSpec, Feature
from vllm.utils.math_utils import round_up

if TYPE_CHECKING:
    from .runtime.decode import KimiK3Kernel

BATCHES = (*range(1, 9), 16, 32, 64)
"""Decode rows the prepared providers serve whatever the deployment captures."""

PREPARED_MAX = BATCHES[-1]
"""The most rows a launch serves; larger steps run vLLM's path."""

KDA_STATE_DTYPES = (torch.bfloat16, torch.float32)
"""The (conv, recurrent) KDA states the prepared KDA cores keep in place."""

MOE = ("situ", 4.0, 25.0, 896, 16, 1, 1, "sigmoid")
"""The one-group sigmoid top-16 SiTU MoE the prepared experts are packed for."""

HIDDEN_SIZE = 7168
"""The hidden size the decode buffers are shaped for."""

MAX_SPECULATIVE_TOKENS = 7
"""The verify kernels' tokens per request, bounded by KdaSplit's barriers."""


def tokens_per_request(vllm_config) -> int:
    """A verify step's tokens per request: 1 + draft tokens, 1 without
    speculation."""
    spec = vllm_config.speculative_config
    return 1 + spec.num_speculative_tokens if spec is not None else 1


def kda_dtypes(vllm_config):
    """(conv, recurrent) KDA state dtypes the deployment's cache config selects."""
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator

    cache = vllm_config.cache_config
    return MambaStateDtypeCalculator.kda_state_dtype(
        vllm_config.model_config.dtype,
        cache.mamba_cache_dtype,
        cache.mamba_ssm_cache_dtype,
    )


def _widths(vllm_config) -> tuple[int, ...]:
    """Every width a launch serves: the prepared batches, plus each captured
    size rounded up to whole requests, up to ``PREPARED_MAX``."""
    q = tokens_per_request(vllm_config)
    sizes = vllm_config.compilation_config.cudagraph_capture_sizes or ()
    verify = {round_up(s, q) for s in sizes}
    return tuple(sorted({*BATCHES, *(m for m in verify if m <= PREPARED_MAX)}))


def _refuse_config(vllm_config) -> list[str]:
    """Every reason this config cannot run the kernels, before load."""
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
    graphs = vllm_config.compilation_config.cudagraph_mode
    reasons = []
    if not current_platform.is_device_capability_family(100):
        reasons.append("needs an SM100/103 GPU")
    if moe != MOE:
        reasons.append(f"needs Kimi K3's SiTU MoE {MOE}, not {moe}")
    if parallel.tensor_parallel_size not in (8, 16):
        reasons.append("needs tensor parallel size 8 or 16")
    if parallel.decode_context_parallel_size not in (
        1,
        parallel.tensor_parallel_size,
    ):
        reasons.append("needs decode context parallel size 1 or the TP size")
    # The decode loop runs in forward; a compiled forward would trace it.
    if vllm_config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE:
        reasons.append("needs a forward that is not torch.compile'd")
    # A launch only ever runs while a FULL decode graph is captured.
    if graphs is None or graphs.decode_mode() != CUDAGraphMode.FULL:
        reasons.append("needs FULL decode CUDA graphs (not enforce-eager)")
    if spec is not None:
        if spec.method != "dspark":
            reasons.append(f"needs the DSpark speculative method, not {spec.method}")
        if spec.enable_adaptive_verification:
            reasons.append("needs adaptive verification off")
        if spec.num_speculative_tokens > MAX_SPECULATIVE_TOKENS:
            reasons.append(f"needs at most {MAX_SPECULATIVE_TOKENS} draft tokens")
        if envs.VLLM_KIMI_K3_AUX_ATTN_RES_STREAM:
            reasons.append("needs the AttnRes aux stream off while speculating")
    # KdaSplit keeps a BF16 conv state and an FP32 recurrent state in place; it
    # maintains neither other cache dtypes nor ReplaySSM / RecoverSSM records.
    if kda_dtypes(vllm_config) != KDA_STATE_DTYPES:
        reasons.append(
            "needs a BF16 KDA conv state and an FP32 recurrent state "
            "(--mamba-cache-dtype)"
        )
    if vllm_config.cache_config.use_replayssm or (
        vllm_config.cache_config.use_kda_recoverssm
    ):
        reasons.append("needs ReplaySSM / RecoverSSM off (--use-replayssm)")
    return reasons


def _refuse_model(backbone) -> list[str]:
    """Every reason the loaded backbone cannot run the kernels."""
    reasons = []
    hidden = getattr(backbone.config, "hidden_size", None)
    if hidden != HIDDEN_SIZE:
        reasons.append(f"needs hidden size {HIDDEN_SIZE}, the model has {hidden}")
    if not getattr(backbone, "num_attn_res_blocks", 0):
        reasons.append("needs the backbone's AttnRes blocks")
    return reasons


def _refuse_caches(vllm_config, kernel: "KimiK3Kernel", caches) -> list[str]:
    """Every reason the prepared cores cannot compile against these caches: a
    KDA layer's states are kept in place, and an MLA layer's KV is indexed as
    one contiguous tensor."""
    reasons = []
    for name, kind in kernel.cache_kinds.items():
        cache = caches.get(name)
        tensors = [cache] if isinstance(cache, torch.Tensor) else list(cache or ())
        if kind == "kda":
            if len(tensors) < 2:
                reasons.append(f"{name}: {len(tensors)} KDA states, needs 2")
                continue
            dtypes = tuple(t.dtype for t in tensors[:2])
            if dtypes != KDA_STATE_DTYPES:
                reasons.append(f"{name}: KDA states {dtypes}, needs {KDA_STATE_DTYPES}")
        elif not tensors:
            reasons.append(f"{name}: no MLA KV cache")
        elif not tensors[0].is_contiguous():
            reasons.append(f"{name}: MLA KV cache is not contiguous")
    return reasons


def _skip_step(vllm_config, kernel: "KimiK3Kernel", width: int) -> str | None:
    """Why this step keeps vLLM's path, or None when the kernels take it."""
    # Every stock kernel has to run before its graph is captured, so a launch
    # only ever replaces one inside a capture: profiling, warm-up and the
    # warm-up run before each capture all run vLLM's path.
    if not torch.cuda.is_current_stream_capturing():
        return "not capturing a CUDA graph"
    if kernel.step_plan(width) is None:
        return "not a decode or uniform verify step the providers serve"
    return None


def _build(vllm_config, model, widths: tuple[int, ...]) -> "KimiK3Kernel":
    from .runtime.decode import KimiK3Kernel

    return KimiK3Kernel(vllm_config, model, widths)


DECODE_SPEC = DecodeSpec["KimiK3Kernel"](
    name="kimi-k3-sm100",
    supports=frozenset({Feature.DECODE_CONTEXT_PARALLEL, Feature.SPECULATIVE_DECODING}),
    widths=_widths,
    refuse_config=_refuse_config,
    refuse_model=_refuse_model,
    refuse_caches=_refuse_caches,
    skip_step=_skip_step,
    build=_build,
)
