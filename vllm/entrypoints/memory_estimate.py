# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Estimate static GPU memory per pipeline stage, without loading weights.

Builds each PP stage's TP-rank-0 worker through the normal startup path, with
meta weights (`load_format="meta"`) on a topology-only process-group world
(`ParallelConfig.fake_distributed`), and plans the KV cache as engine startup
does. Activations, CUDA graphs and communication buffers are not included.
Usage: docs/configuration/conserving_memory.md#estimate-memory-before-launching
"""

import dataclasses
import gc
import json
import os
import sys
from collections.abc import Iterable, Iterator

import torch
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader.meta_loader import host_storages
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder
from vllm.v1.core.kv_cache_planning import (
    _max_memory_usage_bytes_from_groups,
    _pool_bytes_per_block_from_builder,
    _project_kv_cache_groups_to_worker,
    get_global_kv_cache_groups,
)
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.kv_cache_interface import KVCacheGroup, KVCacheLayerConfig
from vllm.v1.worker.worker_base import WorkerWrapperBase


@dataclasses.dataclass
class StageMemoryEstimate:
    """Static GPU memory of one pipeline stage (its TP rank 0)."""

    weight_bytes: int
    """Target and draft model tensors on the GPU, including tensors kept as
    module attributes. Host (pinned or offloaded) tables are excluded."""
    kv_cache_bytes: dict[int, int]
    """Context length -> KV cache one such request needs, with the null block."""


class _StageExecutor(UniProcExecutor):
    """Builds its single worker at a given rank of the fake world."""

    def __init__(self, vllm_config: VllmConfig, rank: int):
        self._stage_rank = rank
        super().__init__(vllm_config)

    def _distributed_args(self) -> tuple[str, int, int]:
        init_method, _, local_rank = super()._distributed_args()
        return init_method, self._stage_rank, local_rank


def _device_tensor_bytes(models: list[nn.Module]) -> int:
    """Bytes of the parameters, buffers and tensor attributes on the GPU."""
    host = set().union(*(host_storages(model) for model in models))
    storages: dict[int, int] = {}
    for module in (module for model in models for module in model.modules()):
        values = [*module._parameters.values(), *module._buffers.values()]
        for value in vars(module).values():
            values.extend(value if isinstance(value, (list, tuple)) else [value])
        for tensor in values:
            if isinstance(tensor, torch.Tensor) and tensor.device.type != "cpu":
                storage = tensor.untyped_storage()
                if storage._cdata not in host:
                    # Tied weights and views share one storage: count it once.
                    storages[storage._cdata] = storage.nbytes()
    return sum(storages.values())


def _model_tensor_bytes(worker: WorkerWrapperBase) -> int:
    models = [worker.get_model(), worker.get_draft_model()]
    return _device_tensor_bytes([model for model in models if model is not None])


def _stage_config(engine_args: EngineArgs) -> VllmConfig:
    # Serving defaults (e.g. max_num_batched_tokens) shape the KV cache need.
    config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    config.parallel_config.fake_distributed = True
    spec_config = config.speculative_config
    if spec_config is not None and spec_config.draft_load_config is not None:
        spec_config.draft_load_config.load_format = "meta"
    return config


def _request_kv_cache_bytes(
    config: VllmConfig,
    groups: list[KVCacheGroup],
    layer_configs: dict[str, KVCacheLayerConfig],
) -> int:
    """KV cache one `max_model_len` request needs on a worker, as the engine's
    admission check counts it, plus the null block the block pool holds back."""
    worker_groups = _project_kv_cache_groups_to_worker(groups, layer_configs)
    if not worker_groups:
        return 0
    builder = get_kv_cache_config_builder(config)
    request = _max_memory_usage_bytes_from_groups(builder, config, worker_groups)
    return request + _pool_bytes_per_block_from_builder(builder, config, worker_groups)


def estimate_static_memory(
    engine_args: EngineArgs, context_lengths: list[int]
) -> list[StageMemoryEstimate]:
    """Estimate every pipeline stage's static device memory, in stage order."""
    engine_args = dataclasses.replace(
        engine_args,
        load_format="meta",
        max_model_len=max(context_lengths),
        # One process stands in for every node; "mp" only has to pass validation.
        distributed_executor_backend="mp",
        nnodes=1,
        node_rank=0,
    )
    config = _stage_config(engine_args)
    parallel = config.parallel_config
    stage_size = parallel.world_size // parallel.pipeline_parallel_size
    layer_configs: list[dict[str, KVCacheLayerConfig]] = []
    layouts: list[list[str]] = []
    weight_bytes: list[int] = []
    for pp_rank in range(parallel.pipeline_parallel_size):
        # Building a worker mutates its config, so each stage gets a fresh one.
        stage_config = config if pp_rank == 0 else _stage_config(engine_args)
        # Keep existing objects out of model loading's full garbage collections.
        gc.freeze()
        try:
            executor = _StageExecutor(stage_config, rank=pp_rank * stage_size)
            try:
                layer_configs.append(executor.get_kv_cache_layer_configs()[0])
                layouts.extend(executor.get_supported_kv_cache_layouts())
                weight_bytes.append(executor.collective_rpc(_model_tensor_bytes)[0])
            finally:
                executor.shutdown()
        finally:
            # Only the process groups must go before the next stage.
            destroy_model_parallel()
            destroy_distributed_environment()
    gc.unfreeze()

    # Stage 0's config carries the block size its attention backend chose.
    with set_current_vllm_config(config):
        all_layers = [lc for stage in layer_configs for lc in stage.values()]
        resolve_kv_cache_layout(config, layouts, all_layers)
        groups = get_global_kv_cache_groups(config, layer_configs)
        config.adjust_dcp_kv_cache_interleave_size(groups)
        estimates: list[StageMemoryEstimate] = []
        for stage, stage_weight_bytes in zip(layer_configs, weight_bytes):
            kv_cache_bytes: dict[int, int] = {}
            for context_length in context_lengths:
                config.model_config.max_model_len = context_length
                kv_cache_bytes[context_length] = _request_kv_cache_bytes(
                    config, groups, stage
                )
            estimates.append(StageMemoryEstimate(stage_weight_bytes, kv_cache_bytes))
    return estimates


def parallel_layouts(nnodes: int, gpus_per_node: int) -> Iterator[dict]:
    """Every TP x PP x DP split of the GPUs (TP within a node), with and
    without expert parallelism."""
    gpus = nnodes * gpus_per_node
    for tp in range(1, gpus_per_node + 1):
        for pp in range(1, gpus // tp + 1):
            if gpus_per_node % tp == 0 and (gpus // tp) % pp == 0:
                for ep in (False, True):
                    yield {
                        "tensor_parallel_size": tp,
                        "pipeline_parallel_size": pp,
                        "data_parallel_size": gpus // tp // pp,
                        "enable_expert_parallel": ep,
                    }


def _estimate_layout(
    engine_args: EngineArgs, context_lengths: list[int], layout: dict
) -> dict:
    try:
        args = dataclasses.replace(engine_args, **layout)
        estimates = estimate_static_memory(args, context_lengths)
    except Exception as error:
        return {**layout, "error": f"{type(error).__name__}: {error}"}
    return {**layout, "stages": [dataclasses.asdict(e) for e in estimates]}


def sweep_static_memory(
    engine_args: EngineArgs,
    context_lengths: list[int],
    layouts: Iterable[dict],
    jobs: int,
) -> Iterator[dict]:
    """Estimate each layout in its own process, `jobs` at once, forked from this
    one so vLLM is imported only once. Yields results as they finish."""
    assert not torch.cuda.is_initialized(), "CUDA does not survive a fork"
    pending = list(layouts)
    running: dict[int, tuple[dict, int]] = {}  # pid -> (layout, result file)
    while pending or running:
        while pending and len(running) < jobs:
            layout = pending.pop(0)
            # A file, unlike a pipe, never blocks a child writing a large result.
            fd = os.memfd_create("memory_estimate")
            if (pid := os.fork()) == 0:
                try:
                    record = _estimate_layout(engine_args, context_lengths, layout)
                    with os.fdopen(fd, "w") as result:
                        json.dump(record, result)
                finally:
                    os._exit(0)
            running[pid] = (layout, fd)
        pid, _ = os.wait()
        layout, fd = running.pop(pid)
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(fd) as result:
            output = result.read()
        yield json.loads(output) if output else {**layout, "error": "process died"}


def main() -> None:
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser(description=__doc__))
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        required=True,
        help="Request lengths to size the KV cache for.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Estimate every TP x PP x DP (x EP) layout of --nnodes x "
        "--gpus-per-node GPUs instead of the given one; prints a JSON line "
        "per layout.",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=8,
        help="With --sweep: GPUs per node; tensor parallelism stays in a node.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="With --sweep: layouts estimated at once, one process each.",
    )
    args = parser.parse_args()
    sys.stdout.flush()
    # Keep stdout for the result; anything else on fd 1 (logs, native
    # libraries) goes to stderr.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w") as result:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        engine_args = EngineArgs.from_cli_args(args)
        if args.sweep:
            layouts = parallel_layouts(args.nnodes, args.gpus_per_node)
            sweep = sweep_static_memory(
                engine_args, args.context_lengths, layouts, args.jobs
            )
            for record in sweep:
                result.write(json.dumps(record) + "\n")
                result.flush()
        else:
            estimates = estimate_static_memory(engine_args, args.context_lengths)
            json.dump([dataclasses.asdict(e) for e in estimates], result)
            result.write("\n")
    # The result is out: skip the ~2 s of interpreter and CUDA teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
