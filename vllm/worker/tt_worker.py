import os
from typing import List, Optional, Tuple
import time

import torch

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size, is_pin_memory_available
from vllm.worker.worker import raise_if_cache_size_invalid
from vllm.worker.tt_model_runner import TTModelRunner
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase,
                                     LoraNotSupportedWorkerBase, WorkerInput)

import ttnn
from ttnn import ReplicateTensorToMesh

logger = init_logger(__name__)


class TTCacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the TT and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)
        # TODO: should be 1 since TP=8
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        # TODO: get num devices from device_config.device (device_mesh)
        # TODO: add get_num_devices to worker
        self.num_kv_heads //= 8 # TP=8, tries to use distributed worker if you give LLM 8 TP

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        # self.attn_backend = get_attn_backend(
        #     model_config.get_num_attention_heads(parallel_config),
        #     self.head_size,
        #     self.num_kv_heads,
        #     model_config.get_sliding_window(),
        #     model_config.dtype,
        #     cache_config.cache_dtype,
        #     self.block_size,
        # )

        # Initialize the cache.
        # List of KV caches. Entry is a list containing K and V tensors.
        self.tt_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device.
        The assumption is that KV cache for a layer is packed into one tensor. 
        We will have a separate tensor for K and V.
        """
        # kv_cache_shape = self.attn_backend.get_kv_cache_shape(
        #     num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        
        # K and V each have the following shape: (num_blocks, num_kv_heads, block_size, head_size)
        kv_cache_shape = (num_blocks, self.num_kv_heads, self.block_size, self.head_size)
        kv_cache: List[torch.Tensor] = []
        num_layers = self.num_attention_layers 
        # num_layers = 1 # TODO: Debugging for 1 layer 
        if device == "cpu":
            for _ in range(num_layers):
                # null block in CpuGpuBlockAllocator requires at least that
                # block to be zeroed-out.
                # Zero-initialize CPU cache
                cache_k = torch.zeros(kv_cache_shape,
                                      dtype=self.dtype,
                                      device=device)
                cache_v = torch.zeros(kv_cache_shape,
                                      dtype=self.dtype,
                                      device=device)
                kv_cache.append([cache_k, cache_v])
        else:
            for _ in range(num_layers):
                cache_k = torch.zeros(kv_cache_shape, dtype=self.dtype)
                cache_v = torch.zeros(kv_cache_shape, dtype=self.dtype)
                
                kv_tt = [ttnn.as_tensor(
                    lp,
                    device=self.device_config.device,
                    # TODO: this could be ShardTensorToMesh, removing need for init to know about TP=8. Could affect other calculations which use self.num_kv_heads, though.
                    mesh_mapper=ReplicateTensorToMesh(self.device_config.device),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat8_b
                ) for lp in (cache_k, cache_v)]
                
                kv_cache.append(kv_tt)
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        raise NotImplementedError
        # for i in range(self.num_attention_layers):
        #     self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
        #                                   src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        raise NotImplementedError
        # for i in range(self.num_attention_layers):
        #     self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
        #                                   src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        raise NotImplementedError
        # self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_heads //= 8 # TODO: Make general without using TP=8?
        num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_attention_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total


class TTWorker(LoraNotSupportedWorkerBase, LocalOrDistributedWorkerBase):
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        is_driver_worker: bool,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker

        assert self.device_config.device_type == "tt"
        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        self.model_runner: TTModelRunner = TTModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config
        )
        
        self.cache_engine: List[TTCacheEngine]
        self.tt_cache: List[List]
        
        self.mesh_device = None  # initialized by init_device

        
    @property
    def do_metadata_broadcast(self) -> bool:
        return False  # TTWorker only supports single-worker execution

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return self.tt_cache

    def init_device(self) -> None:
        # TODO: Add support for devices other than T3K
        self.mesh_device = self._open_t3k_mesh_device()
        self.device_config.device = self.mesh_device
        
        # TODO: Add flag for enabling program cache
        self._enable_program_cache()
        
        # TODO: Add flag for enabling async mode
        self._enable_async_mode()

    def load_model(self):
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available blocks for the TT KV cache and
        swappable CPU KV cache.

        The implementation may run profiling or other heuristics to determine
        the size of caches.

        Returns a Tuple[num_tt_blocks, num_cpu_blocks], where num_tt_blocks
        are blocks that are "active" on the device and can be appended to.
        num_cpu_blocks refers to "swapped" blocks in CPU memory and cannot be
        appended to.
        """
        # TODO: Add proper implementation, ignoring block allocation for now
        # Note: can use --max-num-batched-tokens to set max number of batched tokens per iteration in EngineArgs
        # num_tt_blocks = int(self.scheduler_config.max_model_len / self.cache_config.block_size)
        num_tt_blocks = 501 # TODO: debugging 
        num_cpu_blocks = 0
        return num_tt_blocks, num_cpu_blocks

    def initialize_cache(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        """Initialize the KV cache with the given size in blocks.
        
        - Checks cache size is valid
        - Updates cache_config with num_gpu_blocks and num_cpu_blocks
        - init cache engine
        
        Note that CPU, TPU, and openvino workers don't use standard CacheEngine
        """
        # SKip check, since we're setting num_gpu_blocks much lower than would fit max_model_len
        # raise_if_cache_size_invalid(num_gpu_blocks, self.cache_config.block_size, self.model_config.max_model_len)
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        
        self._init_cache_engine()
        
    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = [
            TTCacheEngine(self.cache_config, self.model_config,
                        self.parallel_config, self.device_config)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.tt_cache = [
            self.cache_engine[ve].tt_cache
            for ve in range(self.parallel_config.pipeline_parallel_size)
        ]
    
    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        raise NotImplementedError

    def prepare_worker_input(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> WorkerInput:
        """
        Prepare the inputs to WorkerBase.execute_worker from an execution
        request. This method may move data to the worker's local device. It is
        not allowed to communicate with other workers or devices.
        """
        virtual_engine = execute_model_req.virtual_engine
        num_steps = execute_model_req.num_steps
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        
        # TODO: Add proper implementation, ignoring block allocation for now
        blocks_to_swap_in = 0
        blocks_to_swap_out = 0
        blocks_to_copy = 0

        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
            num_steps=num_steps,
        )

    def execute_worker(self, worker_input: WorkerInput) -> None:
        """
        Process an execution request.
        
        Appears to do swap_in, swap_out, copy for KV blocks, right before executing the model.
        """
        # TODO: Add proper implementation, ignoring block allocation for now
        pass
    
    # Based on LocalOrDistributedWorkerBase::execute_model, excluding the distributed execution
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[List[SamplerOutput]]:
        """Executes at least one model step on the given sequences, unless no
        sequences are provided."""
        start_time = time.perf_counter()

        inputs = self.prepare_input(execute_model_req)
        if inputs is None:
            return None

        model_input, worker_input, kwargs = inputs
        num_steps = worker_input.num_steps

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        intermediate_tensors = None
        orig_model_execute_time = 0.0
        
        output = self.model_runner.execute_model(
            model_input=model_input,
            kv_caches=self.kv_cache[worker_input.virtual_engine]
            if self.kv_cache is not None else None,
            intermediate_tensors=intermediate_tensors,
            num_steps=num_steps,
            **kwargs,
        )

        model_execute_time = time.perf_counter() - start_time
        
        if (self.observability_config is not None
                and self.observability_config.collect_model_execute_time
                and output is not None):
            for o in output:
                o.model_execute_time = (orig_model_execute_time +
                                        model_execute_time)

        # output is List[SamplerOutput]
        return output
    
    # TT-NN utilities
    
    def _get_devices(self):
        if self.mesh_device:
            devices = self.mesh_device.get_devices()
        else:
            devices = []
            logger.warning("No devices exist")
        return devices
    
    def _get_dispatch_core_type(self):
        dispatch_core_type = ttnn.device.DispatchCoreType.WORKER
        if ("WH_ARCH_YAML" in os.environ) and os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
            dispatch_core_type = ttnn.device.DispatchCoreType.ETH
        return dispatch_core_type
    
    def _open_t3k_mesh_device(self):
        device_ids = [0, 4, 5, 1, 2, 6, 7, 3]
        num_devices_requested = len(device_ids)
        device_params = {}
        
        self.pci_ids = [ttnn.GetPCIeDeviceID(i) for i in device_ids[:num_devices_requested]]

        mesh_device = ttnn.open_mesh_device(
            ttnn.MeshShape(1, num_devices_requested),
            device_ids[:num_devices_requested],
            dispatch_core_type=self._get_dispatch_core_type(),
            **device_params,
        )

        logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created")
        return mesh_device
    
    def _enable_program_cache(self):
        devices = self._get_devices()
        if not devices or len(devices) == 0:
            logger.warning("No devices found to apply program cache to: PROGRAM CACHE DISABLED")
        for dev in devices:
            dev.enable_program_cache()
            
    def _enable_async_mode(self):
        devices = self._get_devices()
        if not devices or len(devices) == 0:
            logger.warning("No devices found to apply async mode to: ASYNC MODE DISABLED")
        for dev in devices:
            dev.enable_async(True)
        
    ## Destructor (used to close devices)
    
    def __del__(self):
        if self.mesh_device:
            devices = self.mesh_device.get_devices()
            
            # Disable program cache
            for dev in devices:
                dev.disable_and_clear_program_cache()
            
            # Disable async mode
            for dev in devices:
                dev.enable_async(False)
            
            # Dump device profiler
            for device in devices:
                ttnn.DumpDeviceProfiler(device)

            # Close devices
            ttnn.close_mesh_device(self.mesh_device)
            del self.mesh_device
        
        if hasattr(super(TTWorker, self), '__del__'):
            super().__del__()