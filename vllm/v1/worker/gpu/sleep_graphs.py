# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, single-GPU graph destruction for level-1 sleep."""

import os
import sys
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.config import CompilationMode, CUDAGraphMode

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def validate_graph_sleep(config: "VllmConfig") -> None:
    """Reject unvalidated resource owners before the first destructive sleep."""
    from vllm.platforms import current_platform

    pc = config.parallel_config
    mc = config.model_config
    cc = config.compilation_config
    supported_models = {"Qwen3ForCausalLM", "Qwen3MoeForCausalLM"}
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
        or pc.world_size != 1
        or pc.data_parallel_size != 1
        or pc.prefill_context_parallel_size != 1
        or pc.decode_context_parallel_size != 1
        or pc.distributed_executor_backend != "uni"
        or pc.worker_cls != "vllm.v1.worker.gpu_worker.Worker"
        or pc.use_ubatching
        or config.scheduler_config.async_scheduling
        or cc.cudagraph_mode != CUDAGraphMode.FULL
        or cc.mode != CompilationMode.NONE
        or cc.compile_sizes
        or config.lora_config is not None
        or config.speculative_config is not None
        or config.kv_transfer_config is not None
        or config.weight_transfer_config is not None
        or config.offload_config.uva.cpu_offload_gb
        or config.offload_config.prefetch.offload_group_size
        or os.environ.get("VLLM_SLEEP_OFFLOAD_CUDA_CONTEXT", "0") == "1"
    ):
        raise ValueError(
            "Graph-discard sleep requires an unquantized Qwen3 text model, "
            "the V2 CUDA runner, one GPU, uni executor, cuMem sleep, FULL "
            "graphs without torch.compile, and synchronous scheduling. "
            "Additional offloaders, adapters, transfers and checkpointing "
            "are not supported."
        )
