# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 specialized decode on NVIDIA Blackwell GPUs, enabled with
``additional_config={"decode_backend": "latency_specialized"}``.

The executor (``runtime/decode.py``) runs the decoder layers of FULL CUDA
graph captures of pure decode steps of 1-32 rows (up to four requests with
the seven-token DSpark draft, whose verify steps are 8 rows a request), with
the low-batch sparse attention core after the KV insert (``attention/``) and
the low-batch FFN half of the MoE layers (``moe/``); every other forward
runs the stock layers.
"""

import torch

import vllm.envs as envs
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# One TP4 rank of MiniMax-M3.1-Flash: what the low-batch FFN's kernels are
# compiled for.
HIDDEN = 6144
EXPERTS = 128
TOP_K = 4
INTERMEDIATE_PER_RANK = 768


def unsupported(vllm_config: VllmConfig) -> list[str]:
    """Every requirement of the specialized decode the configuration misses
    (the fused MoE finalize is checked separately)."""
    from vllm.models.minimax_m3.nvidia.model import _use_mega_moe

    parallel = vllm_config.parallel_config
    compilation = vllm_config.compilation_config
    cudagraph_mode = compilation.cudagraph_mode
    config = vllm_config.model_config.hf_text_config
    tp = parallel.tensor_parallel_size
    shared = getattr(config, "n_shared_experts", 0) or 0
    requirements = {
        "an SM100-family GPU (B200, B300)": current_platform.is_cuda()
        and current_platform.is_device_capability_family(100),
        "Model Runner V2": vllm_config.use_v2_model_runner,
        "no pipeline, data, expert or context parallelism or micro-batching": (
            parallel.pipeline_parallel_size == 1
            and parallel.data_parallel_size == 1
            and not parallel.enable_expert_parallel
            and parallel.decode_context_parallel_size == 1
            and parallel.prefill_context_parallel_size == 1
            and not parallel.use_ubatching
        ),
        f"MoE layers of hidden size {HIDDEN}, {EXPERTS} experts, top-{TOP_K} and "
        f"{INTERMEDIATE_PER_RANK} intermediate columns per rank for the routed "
        "and the shared expert": (
            config.hidden_size == HIDDEN
            and config.num_local_experts == EXPERTS
            and config.num_experts_per_tok == TOP_K
            and config.intermediate_size == INTERMEDIATE_PER_RANK * tp
            and shared == 1
            and not _use_mega_moe(vllm_config)
        ),
        "FULL CUDA graphs for decode batches only (FULL_AND_PIECEWISE or "
        "FULL_DECODE_ONLY)": (
            cudagraph_mode is not None
            and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.mixed_mode() != CUDAGraphMode.FULL
        ),
        "no torch.compile of the model": (
            compilation.mode != CompilationMode.STOCK_TORCH_COMPILE
        ),
        # The low-batch FFN binds weight addresses.
        "no sleep mode": not vllm_config.model_config.enable_sleep_mode,
        "no LoRA or batch invariance": (
            vllm_config.lora_config is None and not envs.VLLM_BATCH_INVARIANT
        ),
    }
    return [need for need, ok in requirements.items() if not ok]


def captures_pure_decode() -> bool:
    """Whether this forward is being captured into a FULL CUDA graph, which
    under the decode-only FULL graphs required above is a pure decode step.
    Model Runner V2 captures FULL graphs with the forward context's mode NONE
    and PIECEWISE ones, which report capturing too, with PIECEWISE."""
    return (
        torch.cuda.is_current_stream_capturing()
        and is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE
    )


def specialized_decode_config(vllm_config: VllmConfig) -> bool:
    """Whether the specialized decode is on (``decode_backend``).

    Raises with every missing requirement when requested. The configuration
    checks give the same answer on every TP rank, and the fused MoE finalize the
    low-batch FFN feeds (NVLink multicast) is decided over the TP group inside
    ``supports_fused_moe_finalize``, so every rank raises alike.
    """
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    if backend == "auto":
        return False
    if backend != "latency_specialized":
        raise ValueError("Supported decode_backend values are auto/latency_specialized")
    from vllm.models.minimax_m3.nvidia.ops.finalize_allreduce_norm import (
        supports_fused_moe_finalize,
    )

    missing = unsupported(vllm_config)
    if not supports_fused_moe_finalize(vllm_config):
        missing.append(
            "tensor parallelism with the fused MoE finalize + all-reduce + norm "
            "(NVLink multicast on every rank)"
        )
    if missing:
        raise ValueError(
            f"decode_backend=latency_specialized needs {'; '.join(missing)}."
        )
    logger.info_once(
        "MiniMax-M3 specialized decode on: in FULL-graph decode steps of 1-32 "
        "rows, the sparse attention layers run the low-batch indexer, sparse "
        "attention and o_proj, and MoE layers that leave their finalize to the "
        "next layer run the low-batch FFN."
    )
    return True
