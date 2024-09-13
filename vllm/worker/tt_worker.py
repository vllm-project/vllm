import os
from typing import List, Optional, Tuple
import time

import torch

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.worker.tt_model_runner import TTModelRunner
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase,
                                     LoraNotSupportedWorkerBase, WorkerInput)

import ttnn

logger = init_logger(__name__)


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
        
        self.mesh_device = None  # initialized by init_device
        self.tt_kv_cache = None
        
    @property
    def do_metadata_broadcast(self) -> bool:
        return False  # TTWorker only supports single-worker execution

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return self.tt_kv_cache

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
        num_tt_blocks = int(self.scheduler_config.max_model_len / self.cache_config.block_size)
        num_cpu_blocks = 0
        return num_tt_blocks, num_cpu_blocks

    def initialize_cache(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        """Initialize the KV cache with the given size in blocks.
        """
        pass  # TODO: Add proper implementation, ignoring block allocation for now
    
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