# SPDX-License-Identifier: Apache-2.0
"""An OpenVINO worker class."""
from typing import Any, Dict, List, Optional, Tuple

import openvino as ov
import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.attention import get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, VllmConfig)
from vllm.distributed import (broadcast_tensor_dict,
                              ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.inputs import INPUT_REGISTRY
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.sequence import ExecuteModelRequest, SequenceGroupMetadata
from vllm.utils import bind_kv_cache
from vllm.worker.openvino_model_runner import OpenVINOModelRunner
from vllm.worker.worker_base import LoRANotSupportedWorkerBase, WorkerBase

logger = init_logger(__name__)


class OpenVINOCacheEngine:
    """Manages the KV cache for OpenVINO backend.

    This class is responsible for initializing and managing CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
        ov_core: ov.Core,
        ov_device: str,
    ) -> None:
        assert device_config.device_type == "openvino"
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        if device_config.device.type == "cpu" and \
            cache_config.cache_dtype == ov.Type.u8:
            # Scale, zero point and quantized data will be stored together.
            # The layout for per token per head:
            # |scale(f32)|zeropoint(f32)|quantized data(u8,idx_1)|quantized data(u8,idx_2)|...|quantized data(u8,idx_head_size)| # noqa: E501
            # so, we have to extend head_size by 8, which is sizeof(float)
            # for scale and sizeof(float) for zeropoint
            self.head_size += 8
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        # Note: In CacheConfig, num_gpu_blocks actual is num_cpu_blocks
        # for OpenVINO backend with a CPU target device, because we want
        # to reuse KV cache management in the scheduler.
        self.num_device_blocks = cache_config.num_gpu_blocks
        self.num_swap_blocks = cache_config.num_cpu_blocks

        # Get attention backend.
        self.attn_backend = get_attn_backend(
            self.head_size,
            self.model_config.dtype,
            self.cache_config.cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
        )

        # Initialize the cache.
        self.kv_cache: List[Tuple[ov.Tensor,
                                  ov.Tensor]] = self._allocate_kv_cache(
                                      self.num_device_blocks, ov_core,
                                      ov_device)

        # Initialize the swap.
        self.swap_cache: List[Tuple[ov.Tensor,
                                    ov.Tensor]] = self._allocate_swap_cache(
                                        self.num_swap_blocks, ov_device)

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        ov_core: ov.Core,
        ov_device: str,
    ) -> List[Tuple[ov.Tensor, ov.Tensor]]:
        """Allocates KV cache."""
        k_block_shape = v_block_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)[1:]
        kv_cache: List[Tuple[ov.Tensor, ov.Tensor]] = []

        if current_platform.is_openvino_cpu():
            for _ in range(self.num_layers):
                key_blocks = ov.Tensor(self.cache_config.cache_dtype,
                                       k_block_shape)
                value_blocks = ov.Tensor(self.cache_config.cache_dtype,
                                         v_block_shape)
                kv_cache.append((key_blocks, value_blocks))
        else:
            # Update key_cache shape:
            k_block_shape = (v_block_shape[0], v_block_shape[1],
                             v_block_shape[3], v_block_shape[2])

            remote_context = ov_core.get_default_context(ov_device)

            for _ in range(self.num_layers):
                key_blocks = \
                    remote_context.create_tensor(self.cache_config.cache_dtype,
                                                 ov.Shape(k_block_shape),
                                                 {})

                value_blocks = \
                    remote_context.create_tensor(self.cache_config.cache_dtype,
                                                 ov.Shape(v_block_shape),
                                                 {})

                kv_cache.append((key_blocks, value_blocks))

        return kv_cache

    def _allocate_swap_cache(
        self,
        num_blocks: int,
        ov_device: str,
    ) -> List[Tuple[ov.Tensor, ov.Tensor]]:
        """Allocates swap cache."""
        k_block_shape = v_block_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)[1:]
        swap_cache: List[Tuple[ov.Tensor, ov.Tensor]] = []

        if num_blocks == 0:
            return swap_cache

        assert not current_platform.is_openvino_cpu(), \
            "CPU device isn't supposed to have swap cache"

        # Update key_cache shape:
        k_block_shape = (v_block_shape[0], v_block_shape[1], v_block_shape[3],
                         v_block_shape[2])

        for _ in range(self.num_layers):
            key_blocks = ov.Tensor(self.cache_config.cache_dtype,
                                   k_block_shape)
            value_blocks = ov.Tensor(self.cache_config.cache_dtype,
                                     v_block_shape)
            swap_cache.append((key_blocks, value_blocks))

        return swap_cache

    def swap_in(self, src_to_dst: List[Tuple[int, int]]) -> None:
        for i in range(self.num_layers):
            for swap_tensor, kv_tensor in zip(self.swap_cache[i],
                                              self.kv_cache[i]):
                self.attn_backend.swap_blocks(swap_tensor, kv_tensor,
                                              src_to_dst)

    def swap_out(self, src_to_dst: List[Tuple[int, int]]) -> None:
        for i in range(self.num_layers):
            for swap_tensor, kv_tensor in zip(self.swap_cache[i],
                                              self.kv_cache[i]):
                self.attn_backend.swap_blocks(kv_tensor, swap_tensor,
                                              src_to_dst)

    def copy(self, src_to_dsts: List[Tuple[int, int]]) -> None:
        if (len(src_to_dsts) > 0):
            self.attn_backend.copy_blocks(self.kv_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        cache_dtype: ov.Type,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        if cache_dtype == ov.Type.u8:
            # Scale, zero point and quantized data will be stored together.
            # The layout for per token per head:
            # |scale(f32)|zeropoint(f32)|quantized data(u8,idx_1)|quantized data(u8,idx_2)|...|quantized data(u8,idx_head_size)| # noqa: E501
            # so, we have to extend head_size by 8, which is sizeof(float)
            # for scale and sizeof(float) for zeropoint
            head_size += 8

        key_cache_block = block_size * num_kv_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = cache_dtype.size
        return dtype_size * total


class OpenVINOWorker(LoRANotSupportedWorkerBase):
    """A worker class that executes the model on OpenVINO backend.

    Each worker is associated with a single OpenVINO device. The worker is
    responsible for maintaining the KV cache and executing the model on the
    OpenVINO backend.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        WorkerBase.__init__(self, vllm_config)
        self.ov_core = ov.Core()
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules

            init_cached_hf_modules()
        self.model_runner = OpenVINOModelRunner(
            self.ov_core,
            vllm_config=self.vllm_config,
            kv_cache_dtype=self.vllm_config.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
        )
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: OpenVINOCacheEngine
        self.kv_cache: List[Tuple[ov.Tensor, ov.Tensor]]

    def init_device(self) -> None:
        self.init_distributed_environment()
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of blocks available for the KV cache.

        This determines how many KV blocks can fit into the configured
        KV cache space.
        """
        # For OpenVINO backend, in case of CPU device, the block number will be
        # calculated based on the openvino_kvcache_space_bytes.
        cache_block_size = self.get_cache_block_size_bytes()
        kvcache_space_bytes = self.cache_config.openvino_kvcache_space_bytes

        if current_platform.is_openvino_cpu():
            num_device_blocks = int(kvcache_space_bytes // cache_block_size)
            num_swap_blocks = 0
        else:
            if kvcache_space_bytes > 0:
                logger.info("KV_CACHE size was explicitly configured via "
                            "VLLM_OPENVINO_KVCACHE_SPACE environment "
                            "variable, ignoring profiling run.")
                kv_cache_size = kvcache_space_bytes
            else:
                try:
                    kv_cache_size = self.profile_run()
                except Exception as err:
                    raise RuntimeError(
                        "The error occurred during profile run. This might be "
                        "due to insufficient GPU memory. Consider decreasing "
                        "`max_model_len` to limit the maximum simultaneously "
                        "processed tokens.") from err

            num_device_blocks = int(kv_cache_size // cache_block_size)
            num_swap_blocks = int(self.cache_config.swap_space_bytes //
                                  cache_block_size)

        return num_device_blocks, num_swap_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache. Swappable CPU memory is only
        supported on GPU.

        For CPU, we use the num_gpu_blocks to
        determine how many non-swappable CPU blocks to allocate.
        """

        num_device_blocks = num_gpu_blocks
        num_swap_blocks = num_cpu_blocks

        if current_platform.is_openvino_cpu():
            assert (num_swap_blocks == 0
                    ), f"{type(self)} does not support swappable cache for CPU"

        self._validate_num_blocks(num_device_blocks)
        self.cache_config.num_gpu_blocks = num_device_blocks
        self.cache_config.num_cpu_blocks = num_swap_blocks

        # Initialize the cache.
        self._init_cache_engine()

    def _validate_num_blocks(self, num_blocks: int) -> None:
        """Raise errors if the num_blocks is invalid."""
        if num_blocks <= 0:
            raise ValueError(
                "No available memory for the cache blocks. "
                "Try increasing `VLLM_OPENVINO_KVCACHE_SPACE` when "
                "initializing the engine.")

        max_seq_len = self.cache_config.block_size * num_blocks
        if self.model_config.max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({self.model_config.max_model_len}) "
                "is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`VLLM_OPENVINO_KVCACHE_SPACE` or decreasing `max_model_len` "
                "when initializing the engine.")

    def _init_cache_engine(self) -> None:
        ov_device = envs.VLLM_OPENVINO_DEVICE
        self.cache_engine = OpenVINOCacheEngine(
            self.cache_config,
            self.model_config,
            self.parallel_config,
            self.device_config,
            self.ov_core,
            ov_device,
        )
        self.kv_cache = self.cache_engine.kv_cache
        bind_kv_cache(self.compilation_config.static_forward_context,
                      [self.kv_cache])
        self.model_runner.block_size = self.cache_engine.block_size

        assert self.kv_cache is not None

        # Populate the cache to warmup the memory
        if current_platform.is_openvino_cpu():
            for key_cache, value_cache in self.kv_cache:
                key_cache.data[:] = 0
                value_cache.data[:] = 0

    def cache_swap_in(self, src_to_dst: List[Tuple[int, int]]) -> None:
        self.cache_engine.swap_in(src_to_dst)

    def cache_swap_out(self, src_to_dst: List[Tuple[int, int]]) -> None:
        self.cache_engine.swap_out(src_to_dst)

    def cache_copy(
        self,
        blocks_to_copy: List[Tuple[int, int]],
    ) -> None:
        self.cache_engine.copy(blocks_to_copy)  # type: ignore

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    @torch.inference_mode()
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> List[SamplerOutput]:
        if execute_model_req is None:
            seq_group_metadata_list = None
        else:
            seq_group_metadata_list = execute_model_req.seq_group_metadata_list

        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            num_seq_groups: int = len(seq_group_metadata_list)
            assert execute_model_req is not None
            blocks_to_copy = execute_model_req.blocks_to_copy
            blocks_to_swap_in = execute_model_req.blocks_to_swap_in
            blocks_to_swap_out = execute_model_req.blocks_to_swap_out
            data: Dict[str, Any] = {
                "num_seq_groups": num_seq_groups,
                "blocks_to_copy": execute_model_req.blocks_to_copy,
                "blocks_to_swap_in": execute_model_req.blocks_to_swap_in,
                "blocks_to_swap_out": execute_model_req.blocks_to_swap_out,
            }
            broadcast_tensor_dict(data, src=0)
        else:
            data = broadcast_tensor_dict(src=0)
            num_seq_groups = data["num_seq_groups"]
            blocks_to_copy = data["blocks_to_copy"]
            blocks_to_swap_in = data["blocks_to_swap_in"]
            blocks_to_swap_out = data["blocks_to_swap_out"]

        if current_platform.is_openvino_cpu():
            assert len(execute_model_req.blocks_to_swap_in) == 0
            assert len(execute_model_req.blocks_to_swap_out) == 0
        else:
            self.cache_swap_in(blocks_to_swap_in)
            self.cache_swap_out(blocks_to_swap_out)

        self.cache_copy(blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return []

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.kv_cache)

        # OpenVINO worker only supports single-step execution.
        return [output]

    def init_distributed_environment(self) -> None:
        """Initialize the distributed environment."""

        parallel_config = self.parallel_config
        rank = self.rank
        distributed_init_method = self.distributed_init_method
        init_distributed_environment(
            world_size=parallel_config.world_size,
            rank=rank,
            distributed_init_method=distributed_init_method,
            backend="gloo",
        )

        # A small all_reduce for warmup.
        torch.distributed.all_reduce(torch.zeros(1).cpu())

        ensure_model_parallel_initialized(
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size,
        )

    def get_cache_block_size_bytes(self) -> int:
        """Return the size in bytes of a single KV cache block."""
        return OpenVINOCacheEngine.get_cache_block_size(
            self.cache_config.block_size,
            self.cache_config.cache_dtype,
            self.model_config,
            self.parallel_config,
        )

    def profile_run(self) -> int:
        ov_device = envs.VLLM_OPENVINO_DEVICE

        assert not current_platform.is_openvino_cpu(), \
            "CPU device isn't supposed to use profile run."

        import openvino.properties.device as device
        import openvino.properties.intel_gpu as intel_gpu

        ov_core = self.ov_core
        cache_config = self.cache_config
        model_config = self.model_config
        parallel_config = self.parallel_config
        device_config = self.device_config
        input_registry = INPUT_REGISTRY
        mm_registry = MULTIMODAL_REGISTRY
        mm_registry.init_mm_limits_per_prompt(model_config)

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        def model_profile_run():
            top_k = model_config.get_vocab_size() - 1
            sampling_params = SamplingParams(top_p=0.99, top_k=top_k)

            max_num_batched_tokens = \
                self.scheduler_config.max_num_batched_tokens
            max_num_seqs = self.scheduler_config.max_num_seqs
            tmp_cache_config = CacheConfig(cache_config.block_size,
                                           cache_config.gpu_memory_utilization,
                                           cache_config.swap_space_bytes,
                                           "auto")
            tmp_cache_config.num_gpu_blocks = 1
            tmp_cache_config.num_cpu_blocks = 0
            tmp_cache_config.cache_dtype = cache_config.cache_dtype

            profiling_cache_engine = OpenVINOCacheEngine(
                tmp_cache_config, model_config, parallel_config, device_config,
                ov_core, ov_device)

            # Profile memory usage with max_num_sequences sequences and the
            # total # number of tokens equal to max_num_batched_tokens.
            seqs: List[SequenceGroupMetadata] = []
            for group_id in range(max_num_seqs):
                seq_len = (max_num_batched_tokens // max_num_seqs +
                           (group_id < max_num_batched_tokens % max_num_seqs))
                block_size = cache_config.block_size
                seq_num_blocks = (seq_len + block_size - 1) // block_size

                dummy_data = input_registry \
                    .dummy_data_for_profiling(model_config,
                                              seq_len,
                                              mm_registry)

                block_tables = [[0] * seq_num_blocks] * max_num_seqs
                seq = SequenceGroupMetadata(
                    request_id=str(group_id),
                    is_prompt=True,
                    seq_data={group_id: dummy_data.seq_data},
                    sampling_params=sampling_params,
                    block_tables=block_tables,
                    lora_request=None,
                    multi_modal_data=dummy_data.multi_modal_data)
                seqs.append(seq)

            self.model_runner.block_size = tmp_cache_config.block_size

            bind_kv_cache(self.compilation_config.static_forward_context,
                          profiling_cache_engine.kv_cache)
            # Run the model with the dummy inputs.
            self.model_runner.execute_model(seqs,
                                            profiling_cache_engine.kv_cache)

            # Explicitly revert bind_kv_cache and delete temporary KV cache
            # manager to free KV cache when real inputs will be passed to OV
            bind_kv_cache(self.compilation_config.static_forward_context, [[
                torch.tensor([])
                for _ in range(len(profiling_cache_engine.kv_cache))
            ]])
            del profiling_cache_engine

            logger.info(
                "Start profiling run with dummy inputs to evaluate "
                "memory usage for %s. It might take a while.", ov_device)

        model_profile_run()

        gpu_device_type = ov_core.get_property(ov_device, device.type)
        memory_statistics = \
            ov_core.get_property(ov_device, intel_gpu.memory_statistics)
        memory_utilization = cache_config.gpu_memory_utilization

        if gpu_device_type == device.Type.INTEGRATED and \
            memory_utilization >= 0.9:
            logger.warning(
                "iGPU is used with high gpu_memory_utilization=%f "
                "value. This may cause low performance due to "
                "occupying the majority of available system "
                "memory. Please consider decreasing "
                "gpu_memory_utilization or explicitly setting "
                "`VLLM_OPENVINO_KVCACHE_SPACE` (GB) environment "
                "variable.", memory_utilization)

        # sum up all used device memory
        device_memory_types = ["cl_mem", "usm_device"]
        used_device_mem = \
            sum(memory_statistics.get(key, 0) for key in device_memory_types)

        if gpu_device_type == device.Type.INTEGRATED:
            used_device_mem += memory_statistics.get("usm_host", 0)

        # there could be unaccounted extra memory reserved by kernels, kept
        # in memory pools, etc
        # therefore, add a threshold to account for this
        used_memory_threshold = 1.1
        used_device_mem *= used_memory_threshold

        total_device_memory = \
            ov_core.get_property(ov_device, intel_gpu.device_total_mem_size)

        def format_memory_size(size) -> str:
            units = ["B", "KB", "MB", "GB"]
            unit_index = 0

            while size > 1024 and unit_index < len(units) - 1:
                size /= 1024
                unit_index += 1

            return f"{size:.2f} {units[unit_index]}"

        total_device_memory_str = \
            format(format_memory_size(total_device_memory))
        used_device_memory_str = \
            format(format_memory_size(used_device_mem))

        logger.info(
            "Total %s memory: %s. "
            "Amount of memory required to run the model with "
            "max_num_batched_tokens=%d: %s.", ov_device,
            total_device_memory_str,
            self.scheduler_config.max_num_batched_tokens,
            used_device_memory_str)

        if used_device_mem >= total_device_memory:
            raise RuntimeError(
                f"The required memory size {used_device_memory_str} for model "
                "is higher than the total available device "
                "memory {total_device_memory_str}. Please consider to "
                "decrease `max_num_batched_tokens` or increase "
                "`gpu_memory_utilization`")

        return total_device_memory * memory_utilization - used_device_mem
