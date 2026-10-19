# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in graph destruction for level-1 sleep (dense/MoE, TP/DP/PP/EP)."""

import os
import sys
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Architectures covered by the graph-discard sleep tests. Anything else must
# be opted in through VLLM_SLEEP_DISCARD_GRAPHS_EXTRA_MODELS and is not
# covered by local validation.
VALIDATED_MODELS = {"Qwen3ForCausalLM", "Qwen3MoeForCausalLM"}


def validate_graph_sleep(config: "VllmConfig") -> None:
    """Reject unvalidated resource owners before the first destructive sleep."""
    from vllm.platforms import current_platform

    pc = config.parallel_config
    mc = config.model_config
    cc = config.compilation_config
    supported_models = VALIDATED_MODELS | envs.VLLM_SLEEP_DISCARD_GRAPHS_EXTRA_MODELS
    if (
        sys.platform != "linux"
        or not current_platform.is_cuda()
        or not envs.VLLM_ENABLE_V1_MULTIPROCESSING
        or not config.use_v2_model_runner
        or not mc.enable_sleep_mode
        or mc.sleep_mode_backend != "cumem"
        or mc.enforce_eager
        or mc.quantization is not None
        or mc.enable_nccl_comm_suspend
        or set(mc.architectures) - supported_models
        or not mc.architectures
        or pc.prefill_context_parallel_size != 1
        or pc.decode_context_parallel_size != 1
        or pc.enable_eplb
        # Expert-parallel all2all groups span every DPxTP rank, so each
        # engine core capturing/discarding on its own idle boundary would
        # desynchronize the cross-engine collectives embedded in the graphs.
        or (pc.enable_expert_parallel and pc.data_parallel_size != 1)
        or pc.distributed_executor_backend not in ("uni", "mp", "ray")
        or pc.worker_cls != "vllm.v1.worker.gpu_worker.Worker"
        or pc.use_ubatching
        or config.scheduler_config.async_scheduling
        or cc.cudagraph_mode not in (CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY)
        or cc.mode != CompilationMode.NONE
        or cc.compile_sizes
        or cc.cudagraph_mm_encoder
        or config.lora_config is not None
        or config.speculative_config is not None
        or config.kv_transfer_config is not None
        or config.weight_transfer_config is not None
        or config.offload_config.uva.cpu_offload_gb
        or config.offload_config.prefetch.offload_group_size
        or os.environ.get("VLLM_SLEEP_OFFLOAD_CUDA_CONTEXT", "0") == "1"
    ):
        raise ValueError(
            "Graph-discard sleep requires an unquantized Qwen3 model, "
            "the V2 CUDA runner, the uni/mp/ray executor, cuMem sleep, FULL "
            "or FULL_DECODE_ONLY graphs without torch.compile, and "
            "synchronous scheduling. "
            "Context parallelism, EPLB and expert parallelism combined with "
            "data parallelism are rejected; tensor, data and pipeline "
            "parallelism and expert parallelism are otherwise supported. "
            "Encoder graphs, additional offloaders, adapters, "
            "transfers and checkpointing are not supported. Unlisted "
            "architectures can be opted in through "
            "VLLM_SLEEP_DISCARD_GRAPHS_EXTRA_MODELS at your own risk."
        )
    untested = set(mc.architectures) - VALIDATED_MODELS
    if untested:
        logger.warning_once(
            "Graph-discard sleep allows %s via "
            "VLLM_SLEEP_DISCARD_GRAPHS_EXTRA_MODELS. These architectures are "
            "not covered by local validation; correctness and memory behavior "
            "are not guaranteed.",
            ", ".join(sorted(untested)),
        )
