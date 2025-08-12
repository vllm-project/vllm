# SPDX-License-Identifier: Apache-2.0
import dataclasses
import math
import os
import time
from typing import List, Optional, Tuple, cast

import torch
import ttnn

from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, VllmConfig)
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType
from vllm.worker.tt_model_runner import TTModelInput, TTModelRunner
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase,
                                     LoRANotSupportedWorkerBase, WorkerBase,
                                     WorkerInput)

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
        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)

        self.num_kv_heads = TTCacheEngine.get_num_kv_heads(
            model_config, parallel_config, device_config)

        self.block_size = cache_config.block_size
        self.num_tt_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Initialize the cache.
        # List of KV caches. Entry is a list containing K and V tensors.
        logger.info("Allocating kv caches")
        self.tt_cache = self._allocate_kv_cache(self.num_tt_blocks,
                                                self.device_config.device_type)
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
        # K and V each have the following shape:
        # (num_blocks, num_kv_heads, block_size, head_size)
        kv_cache_shape = (num_blocks, self.num_kv_heads, self.block_size,
                          self.head_size)
        num_layers = self.num_attention_layers
        if device == "cpu":
            kv_cache: List[torch.Tensor] = []
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
            kv_cache = self.cache_config.tt_allocate_kv_cache(
                kv_cache_shape, self.dtype, num_layers)
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        raise NotImplementedError

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        raise NotImplementedError

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        raise NotImplementedError

    @staticmethod
    def get_num_kv_heads(
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> int:
        '''
        Returns the number of KV heads per attention layer (per device). Makes 
        the assumption that we are tensor parallel by min(number of devices, 
        number of KV heads).
        '''
        data_parallel = 1
        if (model_config.override_tt_config
                and "data_parallel" in model_config.override_tt_config):
            data_parallel = model_config.override_tt_config["data_parallel"]
        num_devices = device_config.device.get_num_devices() // data_parallel
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        # TP = num_devices if num_devices < num_kv_heads
        return num_kv_heads // min(num_devices, num_kv_heads)


class TTWorker(LoRANotSupportedWorkerBase, LocalOrDistributedWorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = True,
    ) -> None:
        WorkerBase.__init__(self, vllm_config=vllm_config)
        self.is_driver_worker = is_driver_worker

        assert self.device_config.device_type == "tt"
        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        # whether to use ttnn tracing for model execution,
        # TODO: make this configurable
        self.trace_mode = True

        self.model_runner: TTModelRunner = TTModelRunner(
            vllm_config=vllm_config,
            trace_mode=self.trace_mode,
        )

        self.cache_engine: TTCacheEngine
        self.tt_cache: List

        # initialized by init_device
        self.mesh_device = None

        # Only used for multi-step execution
        self.cached_model_input: Optional[TTModelInput] = None

    @property
    def do_metadata_broadcast(self) -> bool:
        return False  # TTWorker only supports single-worker execution

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return self.tt_cache

    def init_device(self) -> None:
        self.mesh_device = open_mesh_device(
            self.model_config.override_tt_config, self.trace_mode)
        self.device_config.device = self.mesh_device

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
        # TODO: Add proper implementation which runs profiling on TT devices
        if (("Llama-3.1-8B" in self.model_config.model
             or "Mistral-7B" in self.model_config.model)
                and self.device_config.device.get_num_devices() == 1
                and "wormhole_b0" in ttnn.get_arch_name()
            ):  # Llama8B on N150 and Mistral7B on N150
            max_tokens_all_users = 65536
        elif ("Llama-3.2-90B" in self.model_config.model
              and self.device_config.device.get_num_devices() == 8
              and "wormhole_b0" in ttnn.get_arch_name()):  # Llama90B on WH T3K
            max_tokens_all_users = 65536  # [INFO] avoid OOM for Llama-3.2-90B
        else:
            # Note: includes num vision tokens for multi-modal
            max_tokens_all_users = 131072

        # To fit a max batch with (max_tokens_all_users / max batch) per user,
        # allocate an extra block_size per user since vLLM uses a worst-case
        # heuristic and assumes each touched block will require a new
        # allocation. E.g. batch 32, block 64 needs an extra 2048 tokens.
        max_batch = self.scheduler_config.max_num_seqs
        max_tokens_all_users += self.cache_config.block_size * max_batch

        # For multi-step, to fit (max_tokens_all_users / max batch) per user,
        # allocate an extra num_lookahead_slots (num_scheduler_steps - 1 when
        # not using speculative decoding) per user.
        # E.g. batch 32, num_lookahead_slots 9 needs 288 extra tokens.
        max_tokens_all_users += (self.scheduler_config.num_lookahead_slots *
                                 max_batch)

        num_tt_blocks = math.ceil(max_tokens_all_users /
                                  self.cache_config.block_size)
        num_tt_blocks = int(
            num_tt_blocks *
            1.01)  # Add 1% to account for vLLM's watermark_blocks
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
        # Skip check, since we're setting num_gpu_blocks much lower than would
        # fit max_model_len
        # raise_if_cache_size_invalid(
        #     num_gpu_blocks,
        #     self.cache_config.block_size,
        #     self.model_config.max_model_len
        # )
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._init_cache_engine()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None

        # Get helper function from TT model for allocating the kv cache
        self.cache_config.tt_allocate_kv_cache = (
            self.model_runner.model.allocate_kv_cache)

        self.cache_engine = TTCacheEngine(self.cache_config, self.model_config,
                                          self.parallel_config,
                                          self.device_config)
        self.tt_cache = self.cache_engine.tt_cache

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
        
        Appears to do swap_in, swap_out, copy for KV blocks,
        right before executing the model.
        """
        # TODO: Add proper implementation, ignoring block allocation for now
        pass

    # Based on LocalOrDistributedWorkerBase::execute_model,
    # excluding the distributed execution
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[List[SamplerOutput]]:
        """Executes at least one model step on the given sequences, unless no
        sequences are provided."""
        if execute_model_req is None:
            return None

        start_time = time.perf_counter()

        is_first_multi_step = execute_model_req.is_first_multi_step

        if not self.scheduler_config.is_multi_step or is_first_multi_step:
            inputs = self.prepare_input(execute_model_req)
            if inputs is None:
                return None
            model_input, worker_input, _ = inputs

        if self.scheduler_config.is_multi_step:
            if is_first_multi_step:
                model_input = cast(TTModelInput, model_input)
                self.cached_model_input = model_input
                balance_tokens = []
                metadata_list = execute_model_req.seq_group_metadata_list
                for seq_group_metadata in metadata_list:
                    seq_datas = list(seq_group_metadata.seq_data.values())
                    assert len(seq_datas) == 1, (
                        "Currently only supporting one sequence per "
                        "request group")
                    max_tokens = seq_group_metadata.sampling_params.max_tokens
                    balance_tokens.append(max_tokens -
                                          seq_datas[0].get_output_len())
                balance_tokens = max(balance_tokens)
                worker_input = dataclasses.replace(
                    worker_input,
                    num_steps=min(execute_model_req.num_lookahead_slots + 1,
                                  balance_tokens))
            else:
                assert self.cached_model_input is not None
                model_input = self.cached_model_input
                worker_input = WorkerInput(
                )  # no worker input needed for subsequent steps
            model_input = dataclasses.replace(
                model_input,
                is_first_multi_step=is_first_multi_step,
                is_last_step=execute_model_req.is_last_step)

        num_steps = worker_input.num_steps

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if len(execute_model_req.seq_group_metadata_list) == 0:
            return []

        intermediate_tensors = None
        orig_model_execute_time = 0.0

        output = self.model_runner.execute_model(
            model_input=model_input,
            kv_caches=self.kv_cache if self.kv_cache is not None else None,
            intermediate_tensors=intermediate_tensors,
            num_steps=num_steps,
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

    ## Destructor (used to close devices)

    def __del__(self):
        # Delete model runner first in case there are model arifacts
        del self.model_runner

        if self.mesh_device:
            close_mesh_device(self.mesh_device,
                              self.model_config.override_tt_config)
            del self.mesh_device

        if hasattr(super(), '__del__'):
            super().__del__()  # type: ignore


# TT-NN utilities, also used by V1 TTWorker


def get_dispatch_core_config(override_tt_config):
    dispatch_core_axis: ttnn.DispatchCoreAxis = None
    if (override_tt_config is not None
            and "dispatch_core_axis" in override_tt_config):
        assert override_tt_config["dispatch_core_axis"] in [
            "row", "col"
        ], ("Invalid dispatch_core_axis:"
            f"{override_tt_config['dispatch_core_axis']}. "
            "Expected: row, col.")
        dispatch_core_axis = (ttnn.DispatchCoreAxis.COL
                              if override_tt_config["dispatch_core_axis"]
                              == "col" else ttnn.DispatchCoreAxis.ROW)

    return ttnn.DispatchCoreConfig(axis=dispatch_core_axis)


def get_fabric_config(override_tt_config, num_devices):
    if num_devices == 1:
        # No fabric config for single device
        fabric_config = None
    else:
        # Set the most common value as default
        is_6u = (
            ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY)
        fabric_config = (ttnn.FabricConfig.FABRIC_1D_RING
                         if is_6u else ttnn.FabricConfig.FABRIC_1D)

    # Override fabric_config if specified in override_tt_config
    if (override_tt_config is not None
            and "fabric_config" in override_tt_config):
        fabric_config_str = override_tt_config["fabric_config"]
        fabric_config_map = {
            "DISABLED": ttnn.FabricConfig.DISABLED,
            "FABRIC_1D": ttnn.FabricConfig.FABRIC_1D,
            "FABRIC_1D_RING": ttnn.FabricConfig.FABRIC_1D_RING,
            "FABRIC_2D": ttnn.FabricConfig.FABRIC_2D,
            "CUSTOM": ttnn.FabricConfig.CUSTOM,
        }
        fabric_config = fabric_config_map.get(fabric_config_str)
        assert fabric_config is not None, (
            f"Invalid fabric_config: {fabric_config_str}. "
            f"Expected one of {list(fabric_config_map.keys())}.")
    return fabric_config


# From tt-metal/conftest.py:
# Set fabric config to passed in value
# Do nothing if not set
# Must be called before creating the mesh device
def set_fabric(override_tt_config, num_devices):
    fabric_config = get_fabric_config(override_tt_config, num_devices)
    if fabric_config:
        ttnn.set_fabric_config(fabric_config)


# From tt-metal/conftest.py:
# Reset fabric config to DISABLED if not None, and do nothing otherwise
# Temporarily require previous state to be passed
# in as even setting it to DISABLED might be unstable
# This is to ensure that we don't propagate
# the instability to the rest of CI
def reset_fabric(override_tt_config, num_devices):
    fabric_config = get_fabric_config(override_tt_config, num_devices)
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def device_params_from_override_tt_config(override_tt_config, trace_mode):
    device_params = {}

    if trace_mode:
        # Set the most common value as default, override later
        device_params["trace_region_size"] = 25000000
        if override_tt_config and "trace_region_size" in override_tt_config:
            device_params["trace_region_size"] = override_tt_config[
                "trace_region_size"]

    if override_tt_config and "worker_l1_size" in override_tt_config:
        device_params["worker_l1_size"] = override_tt_config["worker_l1_size"]

    return device_params


def open_mesh_device(override_tt_config, trace_mode):
    num_devices_available = len(ttnn.get_device_ids())
    mesh_grid_dict = {
        "N150": (1, 1),
        "P100": (1, 1),
        "P150": (1, 1),
        "P150x2": (1, 2),
        "N300": (1, 2),
        "P300": (1, 2),
        "N150x4": (1, 4),
        "P150x4": (1, 4),
        "T3K": (1, 8),
        "TG": (8, 4)
    }
    mesh_device_env = os.environ.get("MESH_DEVICE")
    if mesh_device_env is not None:
        assert mesh_device_env in mesh_grid_dict, (
            f"Invalid MESH_DEVICE: {mesh_device_env}")
        mesh_grid = mesh_grid_dict[mesh_device_env]
    else:
        mesh_grid = (1, num_devices_available)

    if mesh_grid[0] * mesh_grid[1] > num_devices_available:
        assert (f"Requested mesh grid shape {mesh_grid} is larger than "
                f"number of available devices {num_devices_available}")

    device_params = device_params_from_override_tt_config(
        override_tt_config, trace_mode)

    # Set fabric before opening the device
    num_devices_requested = mesh_grid[0] * mesh_grid[1]
    set_fabric(override_tt_config, num_devices_requested)

    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(*mesh_grid),
        dispatch_core_config=get_dispatch_core_config(override_tt_config),
        **device_params,
    )
    logger.info("multidevice with %d devices and grid %s is created",
                mesh_device.get_num_devices(), mesh_grid)
    return mesh_device


def close_mesh_device(mesh_device, override_tt_config):
    # Read device profiler (no-op if not profiling with tracy)
    ttnn.ReadDeviceProfiler(mesh_device)

    # Close devices
    num_devices = mesh_device.get_num_devices()
    ttnn.close_mesh_device(mesh_device)

    # Reset fabric
    reset_fabric(override_tt_config, num_devices)
