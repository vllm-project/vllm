import os
from typing import List, Optional, Tuple, Union

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.sequence import ExecuteModelRequest
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, bind_kv_cache, get_dtype_size
from vllm.worker.tpu_model_runner import ExecutionMode, TPUModelRunner
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase,
                                     LoraNotSupportedWorkerBase, WorkerBase,
                                     WorkerInput)

logger = init_logger(__name__)


class TPUWorker(LoraNotSupportedWorkerBase, LocalOrDistributedWorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool,
    ) -> None:
        WorkerBase.__init__(self, vllm_config=vllm_config)
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        assert self.device_config.device_type == "tpu"
        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        self.model_runner: TPUModelRunner = TPUModelRunner(
            vllm_config=vllm_config, is_driver_worker=is_driver_worker)

    def init_device(self) -> None:
        os.environ["PJRT_DEVICE"] = "TPU"
        torch.set_grad_enabled(False)
        torch.set_default_dtype(self.model_config.dtype)

        # NOTE(woosuk): This is just to initialize the TP group and broadcast
        # the input objects on CPU. The all-reduce and all-gather ops on TPU
        # are invoked by `xm.all_reduce` and `xm.all_gather` which use their
        # own context.
        init_distributed_environment(
            world_size=self.parallel_config.world_size,
            rank=self.rank,
            local_rank=self.local_rank,
            distributed_init_method=self.distributed_init_method,
            backend="gloo",
        )
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size)

        # Device initialization should happen after initializing the distributed
        # runtime.
        self.device = xm.xla_device()
        self.device_config.device = self.device

        # Set random seed.
        set_random_seed(self.model_config.seed)
        xm.set_rng_state(self.model_config.seed, self.device)

        # Increase the cache size limit, which is the maximum number of
        # dynamo graphs that can be compiled.
        # NOTE(woosuk): Usually, we compile 10-15 graphs for prefill and
        # 30-40 graphs for decode. 128 is an arbitrary safe number.
        torch._dynamo.config.cache_size_limit = 128
        # Use persistent cache to avoid XLA recompilation.
        # NOTE(woosuk): Set per-rank cache path since different ranks
        # can have slightly different XLA graphs.
        world_size = self.parallel_config.world_size
        rank = xr.global_ordinal()
        per_rank_path = os.path.join(envs.VLLM_XLA_CACHE_PATH,
                                     f"tp{world_size}_rank{rank}")
        xr.initialize_cache(per_rank_path, readonly=False)

    def load_model(self):
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        head_size = self.model_config.get_head_size()
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)

        # use an empty tensor instead of `None`` to force Dynamo to pass
        # it by reference, rather by specializing on the value ``None``.
        # the `dtype` argument does not matter, and we use `float32` as
        # a placeholder (it has wide hardware support).
        kv_caches = [(torch.tensor([], dtype=torch.float32,
                                   device=self.device),
                      torch.tensor([], dtype=torch.float32,
                                   device=self.device))
                     for _ in range(num_layers)]
        bind_kv_cache(self.compilation_config.static_forward_context,
                      [kv_caches])
        self.model_runner._dummy_run(
            batch_size=1,
            seq_len=self.scheduler_config.max_num_batched_tokens,
            kv_caches=kv_caches,
            exec_mode=ExecutionMode.PREFILL,
        )
        # Synchronize before measuring the memory usage.
        xm.wait_device_ops()

        # Get the maximum amount of memory used by the model weights and
        # intermediate activations.
        m = xm.get_memory_info(self.device)
        total_memory_size = m["bytes_limit"]
        profiled = m["peak_bytes_used"]  # Weights + intermediate activations.

        # Calculate the TPU KV cache size based on profiling.
        usable_memory_size = int(total_memory_size *
                                 self.cache_config.gpu_memory_utilization)
        tpu_kv_cache_bytes = max(usable_memory_size - profiled, 0)
        dtype_btyes = get_dtype_size(self.cache_dtype)
        block_size_bytes = (dtype_btyes * self.cache_config.block_size *
                            num_layers * 2 * head_size * num_kv_heads)
        num_tpu_blocks = tpu_kv_cache_bytes // block_size_bytes
        num_tpu_blocks = (num_tpu_blocks // 8) * 8  # Round down to 8.

        # Calculate the CPU KV cache size based on the config.
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             block_size_bytes)
        num_cpu_blocks = (num_cpu_blocks // 8) * 8  # Round down to 8.
        return num_tpu_blocks, num_cpu_blocks

    def initialize_cache(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        self.block_size = self.cache_config.block_size

        dtype = self.cache_dtype
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()

        self.cpu_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.tpu_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        tpu_cache_shape = self.model_runner.attn_backend.get_kv_cache_shape(
            num_gpu_blocks, self.block_size, num_kv_heads, head_size)
        cpu_cache_shape = self.model_runner.attn_backend.get_kv_cache_shape(
            num_cpu_blocks, self.block_size, num_kv_heads, head_size)
        for _ in range(num_layers):
            tpu_k_cache = torch.zeros(tpu_cache_shape,
                                      dtype=dtype,
                                      device=self.device)
            tpu_v_cache = torch.zeros_like(tpu_k_cache)
            self.tpu_cache.append((tpu_k_cache, tpu_v_cache))
            cpu_k_cache = torch.zeros(cpu_cache_shape,
                                      dtype=dtype,
                                      device="cpu")
            cpu_v_cache = torch.zeros_like(cpu_k_cache)
            self.cpu_cache.append((cpu_k_cache, cpu_v_cache))
        bind_kv_cache(self.compilation_config.static_forward_context,
                      [self.tpu_cache])
        self._warmup_model()

    def _warmup_model(self) -> None:
        # FIXME(woosuk): Here we are abusing `enforce_eager` which is defined
        # for CUDA graphs. We should refactor this part.
        if not self.model_config.enforce_eager:
            # Warm up the model with all possible input shapes so that
            # compilation never happens during the actual execution.
            # This may take ~30 mins for the first run and ~20 mins for the
            # subsequent runs.
            # If `enforce_eager` is True, the ahead-of-time compilation is
            # skipped and the compilation happens during the actual execution,
            # which is bad for performance but useful for development.
            self.model_runner.warmup_model(self.tpu_cache)

    def get_cache_block_size_bytes(self) -> int:
        head_size = self.model_config.get_head_size()
        num_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        num_layers = self.model_config.get_num_layers(self.parallel_config)

        key_cache_block = self.cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = get_dtype_size(self.cache_dtype)
        return dtype_size * total

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        # NOTE(woosuk): This assumes virtual_engine == 0, i.e., no pipeline
        # parallelism.
        return [self.tpu_cache]

    def prepare_worker_input(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> WorkerInput:
        virtual_engine = execute_model_req.virtual_engine
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        blocks_to_swap_in = _make_src_to_dst(
            execute_model_req.blocks_to_swap_in, "cpu", self.device)
        blocks_to_swap_out = _make_src_to_dst(
            execute_model_req.blocks_to_swap_out, self.device, "cpu")
        blocks_to_copy = _make_src_to_dst(execute_model_req.blocks_to_copy,
                                          self.device, self.device)
        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
        )

    def execute_worker(self, worker_input: WorkerInput) -> None:
        virtual_engine = worker_input.virtual_engine
        assert virtual_engine == 0
        attn_backend = self.model_runner.attn_backend
        num_layers = self.model_config.get_num_layers(self.parallel_config)

        # Issue cache operations.
        if worker_input.blocks_to_swap_in is not None:
            src_indices, dst_indices = worker_input.blocks_to_swap_in
            if src_indices.numel() > 0:
                # Swap from CPU to TPU.
                for i in range(num_layers):
                    tpu_k_cache, tpu_v_cache = self.tpu_cache[i]
                    cpu_k_cache, cpu_v_cache = self.cpu_cache[i]
                    k = cpu_k_cache[:, src_indices].to(self.device)
                    v = cpu_v_cache[:, src_indices].to(self.device)
                    _insert_kv(k, v, dst_indices, tpu_k_cache, tpu_v_cache)

        if worker_input.blocks_to_swap_out is not None:
            src_indices, dst_indices = worker_input.blocks_to_swap_out
            if src_indices.numel() > 0:
                # Swap from TPU to CPU.
                for i in range(num_layers):
                    tpu_k_cache, tpu_v_cache = self.tpu_cache[i]
                    cpu_k_cache, cpu_v_cache = self.cpu_cache[i]
                    cpu_k_cache[:, dst_indices] = tpu_k_cache[:, src_indices]
                    cpu_v_cache[:, dst_indices] = tpu_v_cache[:, src_indices]

        if worker_input.blocks_to_copy is not None:
            src_indices, dst_indices = worker_input.blocks_to_copy
            if src_indices.numel() > 0:
                attn_backend.copy_blocks(self.tpu_cache,
                                         (src_indices, dst_indices))


def _make_src_to_dst(
    mapping: List[Tuple[int, int]],
    src_device: Union[torch.device, str],
    dst_device: Union[torch.device, str],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not mapping:
        return None

    src_indices = [i for i, _ in mapping]
    dst_indices = [i for _, i in mapping]
    src_indices = torch.tensor(src_indices,
                               device=src_device,
                               dtype=torch.int64)
    dst_indices = torch.tensor(dst_indices,
                               device=dst_device,
                               dtype=torch.int64)
    return src_indices, dst_indices


@torch.compile(backend="openxla")
def _insert_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    tpu_k_cache: torch.Tensor,
    tpu_v_cache: torch.Tensor,
) -> None:
    torch.ops.xla.dynamo_set_buffer_donor_(tpu_k_cache, True)
    torch.ops.xla.dynamo_set_buffer_donor_(tpu_v_cache, True)
    tpu_k_cache[:, indices] = k
    tpu_v_cache[:, indices] = v
