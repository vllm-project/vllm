"""A GPU worker class."""
import gc
import os
from typing import Dict, List, Tuple, Set, Optional

import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, LoRAConfig)
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils.communication_op import (
    broadcast_tensor_dict)
from vllm.model_executor.parallel_utils.custom_all_reduce import init_custom_ar
from vllm.model_executor.parallel_utils.parallel_state import (
    ensure_model_parallel_initialized, get_stage_parallel_group)
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.utils import get_total_num_gpus, WorkerType
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import ModelRunner
from vllm.lora.request import LoRARequest


class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        kv_cache_dtype: Optional[str] = "auto",
        mscclpp_init_method: str = None,
        worker_type: WorkerType = WorkerType.MIXED,
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.mscclpp_init_method = mscclpp_init_method
        self.worker_type = worker_type
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        self.model_runner = ModelRunner(model_config,
                                        parallel_config,
                                        scheduler_config,
                                        device_config,
                                        lora_config=self.lora_config,
                                        kv_cache_dtype=kv_cache_dtype,
                                        is_driver_worker=is_driver_worker)
        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None

        self.kvcache_comm = None
        self.mscclpp_group = None

    def is_prompt_worker(self) -> bool:
        return self.worker_type == WorkerType.PROMPT

    def is_token_worker(self) -> bool:
        return self.worker_type == WorkerType.TOKEN

    def is_mixed_worker(self) -> bool:
        return self.worker_type == WorkerType.MIXED

    def init_model(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_distributed_environment(self.parallel_config, self.rank,
                                     self.distributed_init_method)
        self.init_mscclpp_comm(self.mscclpp_init_method)
        if not self.parallel_config.disable_custom_all_reduce:
            init_custom_ar()


        # Initialize the model.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()
        if self.parallel_config.sep_prompt_token:
            # Populate Sampler with dst_rank as driver worker's rank.
            self.model_runner.model.sampler.set_dst_rank(self.model_runner.driver_rank)

    def init_mscclpp_comm(self, mscclpp_init_method: Optional[str] = None) -> None:
        if mscclpp_init_method is not None:
            try:
                import mscclpp.comm as mscclpp_comm
            except ImportError:
                raise ImportError(
                    "Failed to import MSCCL++ library. Please make sure that "
                    "the MSCCL++ library is installed."
                )

            self.mscclpp_group = mscclpp_comm.CommGroup(
                rank=self.rank,
                size=self.parallel_config.world_size,
                interfaceIpPortTrio=mscclpp_init_method,
            )
            self.mscclpp_conns = None
            self.worker_type = (
                WorkerType.PROMPT
                if self.rank < self.parallel_config.num_prompt_workers
                else WorkerType.TOKEN
            )

            # Set the driver worker rank for prompt and token workers.
            self.model_runner.driver_rank = (
                self.rank // self.parallel_config.num_prompt_workers
            ) * self.parallel_config.num_prompt_workers
            if self.rank == self.model_runner.driver_rank:
                self.is_driver_worker = True
                self.model_runner.is_driver_worker = True

            # Setup up connections.
            corr_worker_rank = (
                self.mscclpp_group.my_rank + self.parallel_config.num_prompt_workers
            ) % self.mscclpp_group.nranks
            transport = self.mscclpp_group.my_ib_device(
                self.mscclpp_group.my_rank % get_total_num_gpus()
            )
            self.mscclpp_conns = self.mscclpp_group.make_connection(
                [corr_worker_rank], transport
            )

    def setup_kvcache_comm(self) -> None:
        # Setup the communication for the KV cache.
        from vllm.utils import MAX_SLOT_IDS
        from vllm.worker.comm_utils import (
            HEAD_TYPES,
            KVCacheCommunicator,
        )
        import mscclpp.comm as mscclpp_comm

        corr_worker_rank = (
            self.mscclpp_group.my_rank + self.parallel_config.num_prompt_workers
        ) % self.mscclpp_group.nranks

        num_layers = self.model_config.get_num_layers(self.parallel_config)

        # Set up proxy service and proxy channels for KV cache communication.
        self.proxy_service = mscclpp_comm.ProxyService()
        self.proxy_service.start_proxy()

        # register KV cache memory with MSCCL++ proxy channel
        memory_ids = [[None, None] for _ in range(num_layers)]
        for layer_id in range(num_layers):
            for head_type in HEAD_TYPES:
                memory_ids[layer_id][head_type] = self.mscclpp_group.register_memory_with_proxy(
                    self.proxy_service,
                    self.gpu_cache[layer_id][head_type],
                    self.mscclpp_conns,
                )
        # register semaphores with MSCCL++ proxy channel
        # one for each sequence
        proxy_channels = [None for _ in range(MAX_SLOT_IDS)]
        device_handles = [None for _ in range(MAX_SLOT_IDS)]
        for sem_id in range(MAX_SLOT_IDS):
            proxy_channels[sem_id] = self.mscclpp_group.register_semaphore_with_proxy(
                self.proxy_service,
                self.mscclpp_conns,
            )[corr_worker_rank]
            device_handles[sem_id] = proxy_channels[sem_id].device_handle().raw

        all_blocks_size = (
            self.gpu_cache[0][0].numel() * self.gpu_cache[0][0].element_size()
        )
        block_size = all_blocks_size // self.gpu_cache[0][0].size(0)

        self.kvcache_comm = KVCacheCommunicator(block_size, device_handles, memory_ids, self.rank, corr_worker_rank)

        # Populate the attention modules with the KV cache communicator.
        self.set_comm_for_attention_modules()

    def set_comm_for_attention_modules(self) -> None:
        attention_modules = list(filter(lambda module: "PagedAttention" in module.__class__.__name__, self.model_runner.model.modules()))
        for i, attention_module in enumerate(attention_modules):
            attention_module.set_kvcache_comm(self.kvcache_comm)
            attention_module.layer_id = i

    def dismantle_kvcache_comm(self) -> None:
        self.proxy_service.stop_proxy()
        del self.proxy_service
        del self.kvcache_comm
        del self.mscclpp_group
        self.unset_comm_for_attention_modules()

    def set_comm_for_attention_modules(self) -> None:
        attention_modules = list(filter(lambda module: "PagedAttention" in module.__class__.__name__, self.model_runner.model.modules()))
        for i, attention_module in enumerate(attention_modules):
            attention_module.set_kvcache_comm(self.kvcache_comm)
            attention_module.layer_id = i

    def unset_comm_for_attention_modules(self) -> None:
        attention_modules = list(filter(lambda module: "PagedAttention" in module.__class__.__name__, self.model_runner.model.modules()))
        for attention_module in attention_modules:
            del attention_module.kvcache_comm

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
        cache_dtype: str,
    ) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model and returns the maximum
        number of GPU and CPU cache blocks that can be allocated.

        Args:
            block_size: The size of the cache block.
            gpu_memory_utilization: The fraction of the total GPU memory to use.
            cpu_swap_space: The size of the CPU swap space in bytes.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = total_gpu_memory - free_gpu_memory

        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, cache_dtype, self.model_config, self.parallel_config)
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache
        self.model_runner.set_block_size(self.cache_engine.block_size)

    def warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def cache_swap(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        cache_events = self.cache_events if issued_cache_op else None

        # Wait for cache operations to finish.
        # TODO(woosuk): Profile swapping overhead and optimize if needed.
        if cache_events is not None:
            for event in cache_events:
                event.wait()

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
        blocks_to_swap_in: Optional[Dict[int, int]] = None,
        blocks_to_swap_out: Optional[Dict[int, int]] = None,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
        blocks_to_nw: Optional[Dict[int, List[int]]] = None,
    ) -> Optional[SamplerOutput]:
        is_prompt = False
        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            is_prompt = seq_group_metadata_list[0].is_prompt
        if self.is_driver_worker and self.should_execute(is_prompt):
            num_seq_groups = len(seq_group_metadata_list)
            assert blocks_to_swap_in is not None
            assert blocks_to_swap_out is not None
            assert blocks_to_copy is not None
            assert blocks_to_nw is not None
            data = {
                "num_seq_groups": num_seq_groups,
                "blocks_to_swap_in": blocks_to_swap_in,
                "blocks_to_swap_out": blocks_to_swap_out,
                "blocks_to_copy": blocks_to_copy,
                "blocks_to_nw": blocks_to_nw,
                "is_prompt": is_prompt,
            }
            broadcast_tensor_dict(data,
                                  src=self.model_runner.driver_rank,
                                  group=get_stage_parallel_group())
        else:
            data = broadcast_tensor_dict(src=self.model_runner.driver_rank,
                                         group=get_stage_parallel_group())
            num_seq_groups = data["num_seq_groups"]
            blocks_to_swap_in = data["blocks_to_swap_in"]
            blocks_to_swap_out = data["blocks_to_swap_out"]
            blocks_to_copy = data["blocks_to_copy"]
            blocks_to_nw = data["blocks_to_nw"]
            is_prompt = data["is_prompt"]

        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return {}

        if len(blocks_to_nw) and self.is_token_worker() and not is_prompt:
            for sem_id in blocks_to_nw:
                self.kvcache_comm.wait(sem_id)

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.gpu_cache, blocks_to_nw)

        if len(blocks_to_nw) and self.is_prompt_worker() and is_prompt:
            for sem_id in blocks_to_nw:
                self.kvcache_comm.signal_and_flush(sem_id)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    def should_execute(self, is_prompt: bool) -> bool:
        return self.is_mixed_worker() or (
            self.is_prompt_worker() and is_prompt) or (
                self.is_token_worker() and not is_prompt)

    def set_gpucache(self):
        from vllm.worker.comm_utils import HEAD_TYPES
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        for layer_id in range(num_layers):
            for head_type in HEAD_TYPES:
                self.gpu_cache[layer_id][head_type][:] = self.rank * (num_layers * len(HEAD_TYPES)) + layer_id * len(HEAD_TYPES) + head_type
        torch.cuda.synchronize()

    def send_recv_kvcache_all(self):
        if self.kvcache_comm is not None:
            num_gpu_blocks = self.cache_config.num_gpu_blocks
            num_layers = self.model_config.get_num_layers(self.parallel_config)
            if self.rank < self.parallel_config.num_prompt_workers:
                for layer_id in range(num_layers):
                    self.kvcache_comm.put(0, layer_id, 0, num_gpu_blocks)
                self.kvcache_comm.signal_and_flush(0)
            else:
                self.kvcache_comm.wait(0)
            torch.cuda.synchronize()

    def check_gpucache(self):
        if self.kvcache_comm is not None:
            from vllm.worker.comm_utils import HEAD_TYPES
            num_prompt_workers = self.parallel_config.num_prompt_workers
            num_layers = self.model_config.get_num_layers(self.parallel_config)
            expected_worker_id = self.rank if self.rank < num_prompt_workers else self.rank - num_prompt_workers
            for layer_id in range(num_layers):
                for head_type in HEAD_TYPES:
                    expected_scalar = expected_worker_id * (num_layers * len(HEAD_TYPES)) + layer_id * len(HEAD_TYPES) + head_type
                    expected_tensor = torch.ones_like(self.gpu_cache[layer_id][head_type]) * expected_scalar
                    assert torch.allclose(self.gpu_cache[layer_id][head_type], expected_tensor)

def init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size,
                                      parallel_config.sep_prompt_token)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")
